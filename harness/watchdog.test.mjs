import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  DEFAULT_POLL_MS,
  DEFAULT_STALL_S,
  LOG_TAIL_BYTES,
  TAIL_LINES,
  initialState,
  isEndingLine,
  logFrame,
  noCounterMarker,
  noteLine,
  readStatsFrame,
  resumeMarker,
  sampleFrame,
  stallMarker,
  statsFrame,
  statusOf,
  statusText,
  tailLines,
  tailPathFor,
  warnStallMarker,
  watch,
} from './watchdog.mjs';
// The evidence rules' helpers, reached through the module so a missing one fails only its own test.
import * as wd from './watchdog.mjs';
import { isolatedEnv } from './test-isolation.mjs';
import { virtualClock } from './virtual-clock.mjs';
import { sampleObject } from './fixtures/old-shape.mjs';
import { statsObject } from './fixtures/runtime-stats.mjs';

const intervalObject = (overrides = {}) => statsObject(overrides);

const here = path.dirname(fileURLToPath(import.meta.url));

function tempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'watchdog-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

const TS = '2026-11-26T12:00:00.000000Z';
// A frame line in the shape both parse-run.mjs and settled-watch.mjs read. `numbered` off is a build before the
// frame counter prefix.
function frameLine(frame, { fps = 30, draws = 5000, dropped = 0, numbered = true } = {}) {
  const prefix = numbered ? `frame ${frame}, ` : '';
  return (
    `${TS}  INFO victory_gpu::host: ${prefix}${fps} fps, 33.3 ms/frame ` +
    '(0.4 preparing: copies 0.2 ms 0.3 MB, textures 0.0 ms 0.0 MB; 0.0 on the GPU; 0.0 copying back 0.0 MB), ' +
    `${draws} draws/frame, dropped ${dropped} without a host pipeline (0 being made) and 0 indirect`
  );
}

// Three heavy lines meet settled-watch.mjs's rule: the harness's own settled signal, reached without a --settled
// watcher running (an open, untimed session).
function armed({ numbered = true, warn = false } = {}) {
  let state = initialState({ warn });
  for (const n of [1, 2, 3]) state = noteLine(state, frameLine(n, { numbered }), { now: n * 1000 });
  return state;
}

test('the watcher arms on the settled rule, not before it', () => {
  let state = initialState();
  state = noteLine(state, frameLine(1), { now: 1000 });
  state = noteLine(state, frameLine(2), { now: 2000 });
  assert.equal(state.armed, false, 'two settled frame lines are not the settle');
  assert.equal(statusText(statusOf(state, { now: 60_000, stallS: 1 })), 'not-armed: not settled');

  state = noteLine(state, frameLine(3), { now: 3000 });
  assert.equal(state.armed, true);
  assert.equal(state.settle.settledAt !== null, true);
});

test('an explicit harness settle line arms an untimed session too', () => {
  let state = noteLine(initialState(), 'harness: settled at 5 s (waiting 0, 30.0 fps)', { now: 5000 });
  assert.equal(state.armed, true);
  state = sampleFrame(state, { at: 6000, frame: 12 });
  assert.equal(statusText(statusOf(state, { now: 6000, stallS: 45 })), 'ok');
  // A stall needs a read after the window showing the same counter; no read at all cannot tell.
  assert.equal(statusOf(state, { now: 6000 + 45_000, stallS: 45 }).status, 'warning');
  state = sampleFrame(state, { at: 6000 + 45_000, frame: 12 });
  assert.equal(statusOf(state, { now: 6000 + 45_000, stallS: 45 }).status, 'stalled');

  // A counter known before the arm, with no change after it, counts from the arm rather than never stalling.
  let unsampled = sampleFrame(initialState(), { at: 1000, frame: 7 });
  unsampled = noteLine(unsampled, 'harness: settled at 5 s (waiting 0, 30.0 fps)', { now: 5000 });
  unsampled = noteLine(unsampled, 'harness: a later line, read after the window', { now: 5000 + 45_000 });
  assert.equal(statusOf(unsampled, { now: 5000 + 45_000, stallS: 45 }).status, 'stalled');
});

test('an advancing counter is ok, and a still one is stalled after the threshold', () => {
  let state = armed();
  state = sampleFrame(state, { at: 4000, frame: 4 });
  assert.equal(statusText(statusOf(state, { now: 4000, stallS: 45 })), 'ok');
  state = sampleFrame(state, { at: 5000, frame: 5 });
  state = sampleFrame(state, { at: 5000 + 44_000, frame: 5 });
  assert.equal(statusOf(state, { now: 5000 + 44_000, stallS: 45 }).status, 'ok', '44 s is under the threshold');
  state = sampleFrame(state, { at: 5000 + 45_000, frame: 5 });
  const { status, evidence, frame, seconds } = statusOf(state, { now: 5000 + 45_000, stallS: 45 });
  assert.deepEqual({ status, evidence, frame, seconds }, { status: 'stalled', evidence: 'log', frame: 5, seconds: 45 });
  assert.equal(DEFAULT_STALL_S, 45);
});

test('a counter that restarts lower is an advance, not a stall', () => {
  let state = armed();
  state = sampleFrame(state, { at: 4000, frame: 900 });
  state = sampleFrame(state, { at: 9000, frame: 3 });
  assert.equal(statusOf(state, { now: 9000 + 44_000, stallS: 45 }).status, 'ok');
  state = sampleFrame(state, { at: 9000 + 45_000, frame: 3 });
  const stalled = statusOf(state, { now: 9000 + 45_000, stallS: 45 });
  assert.equal(stalled.status, 'stalled');
  assert.equal(stalled.frame, 3, 'the restarted counter is the one reported');
});

test('the end of the run disarms the watcher', () => {
  let state = armed();
  state = sampleFrame(state, { at: 4000, frame: 4 });
  state = noteLine(state, 'harness: victory exit=0', { now: 5000 });
  assert.equal(state.ended, true);
  assert.equal(state.armed, false);
  assert.equal(statusText(statusOf(state, { now: 999_999, stallS: 1 })), 'not-armed: run ending');
  for (const line of [
    'harness: ended by --heavy-seconds after 30 s of heavy scene',
    'harness: ended by --settled-timeout after 600 s: the scene never settled',
    'harness: ended: disk below 5 GB',
  ]) {
    assert.equal(noteLine(armed(), line, { now: 5000 }).ended, true, line);
  }
});

test('a frame line without a number has no counter, and the watcher says so', () => {
  const text = [frameLine(1, { numbered: false }), frameLine(2, { numbered: false })].join('\n');
  assert.deepEqual(logFrame(text), { frame: null, sawFrameLine: true });
  assert.deepEqual(logFrame('not a frame line'), { frame: null, sawFrameLine: false });

  const state = armed({ numbered: false });
  assert.equal(state.noFrameNumber, true);
  const result = statusOf(state, { now: 60_000, stallS: 1 });
  assert.equal(result.status, 'not-armed');
  assert.equal(statusText(result), 'not-armed: no frame counter');
  assert.equal(noCounterMarker(result), 'harness: watchdog: not-armed: no frame counter');
  // The watcher's own note is not an end; writing it must not disarm the watcher.
  assert.equal(isEndingLine('harness: watchdog: not-armed: no frame counter'), false);
  assert.equal(noteLine(state, 'harness: watchdog: not-armed: no frame counter', { now: 61_000 }).armed, true);
});

test('the newest frame line gives the counter', () => {
  assert.equal(logFrame([frameLine(5), frameLine(12), 'a later line'].join('\n')).frame, 12);
  assert.equal(logFrame(frameLine(77)).frame, 77);
  assert.equal(LOG_TAIL_BYTES >= 65536, true);
});

test('a missing or half-written stats file is no sample, never a stopped counter', (t) => {
  assert.equal(statsFrame('{"frame": 12, "fps": 30}'), 12);
  assert.equal(statsFrame('{"frame": 12'), null, 'a half-written file');
  assert.equal(statsFrame(''), null);
  assert.equal(statsFrame('null'), null);
  assert.equal(statsFrame('{"frame": "12"}'), null);
  assert.equal(statsFrame('{"frame": null}'), null);

  const dir = tempDir(t);
  assert.equal(readStatsFrame(path.join(dir, 'absent.json')), null);
  const file = path.join(dir, 'stats.json');
  fs.writeFileSync(file, '{"frame": 4');
  assert.equal(readStatsFrame(file), null);
  fs.writeFileSync(file, '{"frame": 4, "fps": 30}');
  assert.equal(readStatsFrame(file), 4);

  // A torn read is no sample, and no sample is not a stopped counter: a healthy game whose stats file cannot be read
  // from 200 s to 260 s (the validator's case 7, generated here) is never stalled, and the gap is a warning.
  let probe = armedStats();
  for (let second = 0; second <= 600; second += 1) {
    const at = 4000 + second * 1000;
    // The validator's form: an unreadable second gives no sample at all.
    if (second < 200 || second >= 260) probe = sampleFrame(probe, { at, frame: 1000 + 30 * second });
    assert.notEqual(statusOf(probe, { now: at, stallS: 45 }).status, 'stalled', `no stall at ${second} s`);
  }
  // The watcher's own form: each failed read is counted, and the gap is "cannot tell".
  let state = armedStats();
  const kinds = new Set();
  for (let second = 0; second <= 600; second += 1) {
    const at = 4000 + second * 1000;
    const read = second >= 200 && second < 260 ? { frame: null, error: 'unparsable' } : { frame: 1000 + 30 * second, mtimeMs: at };
    state = wd.sampleStats(state, { at, read });
    const result = statusOf(state, { now: at, stallS: 45 });
    assert.notEqual(result.status, 'stalled', `no stall at ${second} s`);
    if (result.status === 'warning') kinds.add(result.kind);
  }
  assert.deepEqual([...kinds], ['cannot-tell']);
  assert.equal(state.statsFailures, 60);
});

// A watcher on a --live run, armed by three frame lines read at 1, 2 and 3 s.
// `base` puts the watcher's clock on another epoch (the interval shape's own unix_ms).
function armedStats({ warn = false, base = 0 } = {}) {
  let state = initialState({ source: 'stats', warn });
  for (const n of [1, 2, 3]) state = noteLine(state, frameLine(n), { now: base + n * 1000 });
  return state;
}

test('a stats file that stops changing while the log shows frames is a warning, never a stall (audit: writes failing ended a healthy run)', () => {
  let state = armedStats();
  state = sampleFrame(state, { at: 4000, frame: 40 });
  // The runtime's stats writes fail from here on, so the file keeps frame 40; the game keeps presenting, and its
  // log keeps its frame lines about every 5 s.
  for (let at = 9000; at <= 49_000; at += 5000) {
    state = sampleFrame(state, { at, frame: 40 });
    state = noteLine(state, frameLine(at / 100), { now: at });
  }
  const result = statusOf(state, { now: 4000 + 45_000, stallS: 45 });
  assert.notEqual(result.status, 'stalled', 'a stats file that stopped being written is not a frozen game');
  assert.equal(result.status, 'warning');
  assert.equal(result.kind, 'stats-not-advancing');
  assert.equal(result.frame, 40);
});

test('a still stats counter with no evidence from the log either way is "cannot tell", a warning', () => {
  let state = armedStats();
  state = sampleFrame(state, { at: 4000, frame: 40 });
  // Nothing has read the log since the arm: no frame line, and no read showing their absence.
  const result = statusOf(state, { now: 4000 + 60_000, stallS: 45 });
  assert.notEqual(result.status, 'stalled');
  assert.equal(result.status, 'warning');
  assert.equal(result.kind, 'cannot-tell');
});

test('a still stats counter and a log read after the window with no frame line in it is a stall', () => {
  let state = armedStats();
  state = sampleFrame(state, { at: 4000, frame: 40 });
  state = noteLine(state, 'harness: some other line', { now: 48_000 });
  assert.equal(statusOf(state, { now: 48_000, stallS: 45 }).status, 'ok', '44 s is under the window');
  state = noteLine(state, 'harness: some other line', { now: 49_000 });
  const result = statusOf(state, { now: 49_000, stallS: 45 });
  assert.equal(result.status, 'stalled');
  assert.equal(result.evidence, 'log');
  assert.equal(result.frame, 40);
});

test('a stats file still being rewritten whose frame counter does not move is a stall', () => {
  let state = armedStats();
  state = wd.sampleStats(state, { at: 4000, read: { frame: 40, mtimeMs: 1 } });
  for (let at = 10_000; at <= 52_000; at += 6000) {
    state = wd.sampleStats(state, { at, read: { frame: 40, mtimeMs: at } });
  }
  const result = statusOf(state, { now: 52_000, stallS: 45 });
  assert.equal(result.status, 'stalled');
  assert.equal(result.evidence, 'stats');

  // A rewrite is seen by its text too: a file whose time does not move but whose contents do (a timestamp field,
  // whatever its name) is being written.
  let stamped = wd.sampleStats(armedStats(), { at: 4000, read: { frame: 40, mtimeMs: 1, text: '{"frame":40,"t":1}' } });
  stamped = wd.sampleStats(stamped, { at: 50_000, read: { frame: 40, mtimeMs: 1, text: '{"frame":40,"t":2}' } });
  assert.equal(statusOf(stamped, { now: 52_000, stallS: 45 }).evidence, 'stats');

  // The same file, but the log shows frames: the instruments disagree, so the watcher warns instead of ending.
  const shown = noteLine(state, frameLine(520), { now: 51_000 });
  const disagree = statusOf(shown, { now: 52_000, stallS: 45 });
  assert.equal(disagree.status, 'warning');
  assert.equal(disagree.kind, 'stats-not-advancing');
});

// The runtime's object (fixtures/runtime-stats.mjs): `seq` is the object's identity and `unix_ms` its own wall clock. The runtime
// writes an object only at a present, so a pause is no new seq and then one object holding the whole gap.
const UNIX = 1_795_000_000_000;
function intervalRead(seq, frame, at, { mtimeMs = at, frames = 30, wallMs = 1000, list = null, unixMs = at } = {}) {
  const object = intervalObject({
    seq,
    unix_ms: UNIX + unixMs,
    frame,
    interval: { first_frame: frame - frames + 1, last_frame: frame, frames, wall_ms: wallMs },
    frame_ms: wallMs / frames,
    frame_ms_list: list ?? Array.from({ length: frames }, () => wallMs / frames),
  });
  const text = `${JSON.stringify(object)}\n`;
  return { frame, seq, unixMs: UNIX + unixMs, shape: 'interval', mtimeMs, text };
}
// The watcher's clock on the same epoch as the objects' own.
const at = (ms) => UNIX + ms;

test('readStats reads both shapes: seq, unix_ms and the shape of an interval object, null for the older sample', (t) => {
  const dir = tempDir(t);
  const file = path.join(dir, 'stats.json');
  fs.writeFileSync(file, JSON.stringify(intervalObject()));
  const fresh = wd.readStats(file);
  assert.equal(fresh.frame, 1234);
  assert.equal(fresh.seq, 8);
  assert.equal(fresh.unixMs, 1800000008000);
  assert.equal(fresh.shape, 'interval');
  fs.writeFileSync(file, JSON.stringify(sampleObject()));
  const older = wd.readStats(file);
  assert.equal(older.frame, 2048);
  assert.equal(older.seq, null);
  assert.equal(older.unixMs, null);
  assert.equal(older.shape, 'sample');
});

test('a repeated seq is neither progress nor a rewrite, even when the file time moves', () => {
  let state = armedStats({ base: UNIX });
  state = wd.sampleStats(state, { at: at(4000), read: intervalRead(412, 4217, 4000) });
  // The same object, its file touched every 6 s (a copy, an indexer): the old rule read a moved time as a write.
  for (let ms = 10_000; ms <= 52_000; ms += 6000) {
    state = wd.sampleStats(state, { at: at(ms), read: { ...intervalRead(412, 4217, 4000), mtimeMs: ms } });
  }
  const result = statusOf(state, { now: at(52_000), stallS: 45 });
  assert.notEqual(result.status, 'stalled', 'one object read again is no evidence that frames stopped');
  assert.equal(result.status, 'warning');
  assert.equal(result.kind, 'cannot-tell');
});

test('a new seq is progress, as a changed frame is for the older shape', () => {
  let state = armedStats({ base: UNIX });
  // The frame number is the same in each (as if the counter were stuck); the seq moves, so the runtime presented.
  for (let ms = 4000, seq = 400; ms <= 60_000; ms += 1000, seq += 1) {
    state = wd.sampleStats(state, { at: at(ms), read: intervalRead(seq, 40, ms) });
  }
  assert.equal(statusOf(state, { now: at(60_000), stallS: 45 }).status, 'ok');
  assert.equal(state.seq, 456);
});

test('a seq jump counts the objects missed, and the verdict says how many', () => {
  let state = armedStats({ base: UNIX });
  state = wd.sampleStats(state, { at: at(4000), read: intervalRead(10, 300, 4000) });
  state = wd.sampleStats(state, { at: at(5000), read: intervalRead(13, 390, 5000) });
  state = wd.sampleStats(state, { at: at(6000), read: intervalRead(14, 420, 6000) });
  assert.equal(state.statsMissed, 2);
  const result = statusOf(state, { now: at(6000), stallS: 45 });
  assert.match(wd.verdictText(result), /stats file: 0 failed reads of 3, 2 objects missed/);
  // The older shape has no seq: nothing is counted, and the verdict says nothing about it.
  const older = wd.sampleStats(armedStats(), { at: 4000, read: { frame: 40, mtimeMs: 1 } });
  assert.doesNotMatch(wd.verdictText(statusOf(older, { now: 4000, stallS: 45 })), /missed/);
});

test('freshness by unix_ms: a still interval object is aged by its own clock in the warning', () => {
  let state = armedStats({ base: UNIX });
  // Written at 4 s by its own clock, read every second after it; nothing reads the log.
  state = wd.sampleStats(state, { at: at(4000), read: intervalRead(412, 4217, 4000) });
  for (let ms = 5000; ms <= 64_000; ms += 1000) {
    state = wd.sampleStats(state, { at: at(ms), read: intervalRead(412, 4217, 4000) });
  }
  const result = statusOf(state, { now: at(64_000), stallS: 45 });
  assert.equal(result.status, 'warning');
  assert.match(result.reason, /the newest stats object \(seq 412\) was written 60 s ago by its own clock/);
});

test('a pause in the interval shape: no new seq stalls on the log evidence, and the one object after it resumes', () => {
  // Objects about once a second and a frame line every 5 s, then 60 s with no present: no object, no frame line.
  let state = armedStats({ warn: true, base: UNIX });
  let seq = 100;
  let frame = 3000;
  for (let ms = 4000; ms <= 20_000; ms += 1000) {
    seq += 1;
    frame += 30;
    state = wd.sampleStats(state, { at: at(ms), read: intervalRead(seq, frame, ms) });
    if (ms % 5000 === 0) state = noteLine(state, frameLine(frame), { now: at(ms) });
    state = wd.noteLogRead(state, { at: at(ms) });
  }
  const frozen = intervalRead(seq, frame, 20_000);
  for (let ms = 21_000; ms <= 80_000; ms += 1000) {
    state = wd.sampleStats(state, { at: at(ms), read: frozen });
    state = wd.noteLogRead(state, { at: at(ms) });
  }
  const stalled = statusOf(state, { now: at(80_000), stallS: 45 });
  assert.equal(stalled.status, 'stalled');
  assert.equal(stalled.evidence, 'log', 'the log, read with no frame line in it; the stats file was not rewritten');
  // The caller records the stall it announced, then the next object holds the whole gap: one 60 s frame and 29 more.
  state = { ...state, stall: { frame, seconds: 45, since: state.advancedAt } };
  const resume = intervalRead(seq + 1, frame + 30, 81_000, {
    frames: 30,
    wallMs: 60_000 + 29 * 33,
    list: [60_000, ...Array.from({ length: 29 }, () => 33)],
  });
  state = wd.sampleStats(state, { at: at(81_000), read: resume });
  assert.deepEqual(state.resume, { frame: frame + 30, seconds: 61 });
  assert.equal(state.stall, null);
  assert.equal(statusOf(state, { now: at(81_000), stallS: 45 }).status, 'ok');
});

test('every failed read of the stats file is counted once armed, and the verdict says how many', (t) => {
  const dir = tempDir(t);
  const file = path.join(dir, 'stats.json');
  assert.deepEqual(wd.readStats(path.join(dir, 'absent.json')), { frame: null, error: 'missing' });
  fs.writeFileSync(file, '{"frame": 4');
  assert.deepEqual(wd.readStats(file), { frame: null, error: 'unparsable' });
  fs.writeFileSync(file, '{"fps": 30}');
  assert.deepEqual(wd.readStats(file), { frame: null, error: 'no frame' });
  fs.writeFileSync(file, '{"frame": 4, "fps": 30}');
  const good = wd.readStats(file);
  assert.equal(good.frame, 4);
  assert.equal(Number.isFinite(good.mtimeMs), true);

  // Before the arm the file may not exist yet: not a failure.
  const early = wd.sampleStats(initialState({ source: 'stats' }), { at: 500, read: { frame: null, error: 'missing' } });
  assert.equal(early.statsFailures, 0);
  let state = wd.sampleStats(armedStats(), { at: 4000, read: { frame: 40, mtimeMs: 1 } });
  for (const error of ['unparsable', 'missing', 'unparsable', 'no frame']) {
    state = wd.sampleStats(state, { at: 5000, read: { frame: null, error } });
  }
  assert.equal(state.statsFailures, 4, 'each failed read, not only the first');
  assert.equal(state.statsReads, 5);
  const result = statusOf(state, { now: 4000 + 60_000, stallS: 45 });
  assert.equal(result.statsFailures, 4);
  assert.match(wd.verdictText(result), /stats file: 4 failed reads of 5/);
  assert.match(wd.warningMarker(result), /^harness: watchdog: warning: .*stats file: 4 failed reads of 5.*; NOT ending the run$/);
});

test('the stall line and the tail file are named as the log expects', (t) => {
  assert.equal(
    stallMarker(4321, 47),
    'harness: watchdog: the frame counter stopped at frame 4321 for 47 s; ending the run',
  );
  assert.equal(TAIL_LINES, 300);
  assert.equal(tailPathFor('logs/20260101-120000-run.log'), 'logs/20260101-120000-run-stall-tail.log');
  assert.equal(tailPathFor('logs/run'), 'logs/run-stall-tail.log');

  const dir = tempDir(t);
  const file = path.join(dir, 'run.log');
  fs.writeFileSync(file, Array.from({ length: 400 }, (_, i) => `line ${i + 1}`).join('\n') + '\n');
  const tail = tailLines(file);
  const rows = tail.trimEnd().split(/\r?\n/);
  assert.equal(rows.length, 300);
  assert.equal(rows[0], 'line 101');
  assert.equal(rows[299], 'line 400');
  assert.equal(tailLines(file, 5), 'line 396\nline 397\nline 398\nline 399\nline 400\n');
  assert.equal(tailLines(path.join(dir, 'absent.log')), '');
});

test('warn mode names its stall and its resume, and counts the stall it saw', () => {
  assert.equal(
    warnStallMarker(4321, 47),
    'harness: watchdog: the frame counter stopped at frame 4321 for 47 s; NOT ending the run (warn mode)',
  );
  assert.equal(stallMarker(4321, 47, { warn: true }), warnStallMarker(4321, 47));
  assert.equal(
    resumeMarker(4330, 47),
    'harness: watchdog: frames resumed at frame 4330 after 47 s',
  );

  let state = armed({ warn: true });
  state = sampleFrame(state, { at: 4000, frame: 4 });
  // The watcher's tick records the stall it just announced; the next sample is the tick after it.
  state = {
    ...state,
    stalls: 1,
    longestS: 45,
    stall: { frame: 4, seconds: 45, since: 4000 },
  };
  state = sampleFrame(state, { at: 50_000, frame: 4 });
  assert.equal(statusOf(state, { now: 50_000, stallS: 45 }).status, 'stalled');
  const resumed = sampleFrame(state, { at: 51_000, frame: 5 });
  assert.deepEqual(resumed.resume, { frame: 5, seconds: 47 });
  assert.equal(resumed.stall, null);
  assert.equal(resumed.advancedAt, 51_000);
  assert.equal(resumed.longestS, 47);
  assert.equal(resumed.stalls, 1);
  // A warning watcher's state says which mode it is in; the default is still the ending watcher.
  assert.equal(initialState({ warn: true }).warn, true);
  assert.equal(initialState().warn, false);
});

test('the run-world settings are read, echoed, and start the watcher in either mode', () => {
  const text = fs.readFileSync(path.join(here, 'run-world.sh'), 'utf8');
  // The default is decided after the arguments and routes, because a hold timer makes a run timed.
  assert.match(text, /WATCHDOG=\$\{WATCHDOG-\}/);
  assert.match(text, /if \[ -z "\$WATCHDOG" \]; then/);
  assert.match(text, /\[ -n "\$HEAVY_SECONDS" \] \|\| \[ -n "\$SETTLED_SECONDS" \]/);
  assert.match(text, /WATCHDOG=1\n  else\n    WATCHDOG=warn/);
  assert.match(text, /warn\|warn-only\) WATCHDOG=warn/);
  assert.match(text, /WATCHDOG_STALL_S=\$\{WATCHDOG_STALL_S:-45\}/);
  assert.match(text, /watchdog-stall-s=\$WATCHDOG_STALL_S/);
  assert.match(text, /WD_ARGS=\(--stall-seconds "\$WATCHDOG_STALL_S"\)/);
  assert.match(text, /\[ "\$WATCHDOG" = warn \] && WD_ARGS\+=\(--warn\)/);
  assert.match(text, /WD_ARGS\+=\(--stats "\$LIVE_STATS"\)/);
  assert.match(text, /node "\$TOOLS\/watchdog\.mjs" "\$LOG" "\$\{WD_ARGS\[@\]\}"/);
  const start = text.indexOf('watchdog.mjs');
  const summary = text.indexOf('summarize-run.mjs');
  assert.ok(start > 0 && summary > start, 'the watcher starts long before the summary is made');
  assert.ok(text.indexOf('if [ -n "$WATCHDOG_PID" ]') > start, 'it is stopped before the summary');
});

// The fake run, in this process on a virtual clock: it writes the settled scene's log and advances a stats-file
// counter every 50 ms for 400 ms before leaving it standing (`failing`: it then leaves the file torn, so every read
// fails, while the log keeps gaining a frame line every 200 ms: a healthy run whose stats writes fail). The game the
// watcher may end is a separate process that only stays alive (`standIn`), so nothing a test asserts depends on how
// soon a loaded machine schedules a child: the hosted runner's failure. The PowerShell stub below stands in for
// run-process.ps1's strict selection: it prints the stand-in's pid and nothing else, so the watcher can only end the
// test's own process.
const FAKE_LOG = '20260101-120000-run.log';
const FAKE_TS = '2026-11-26T12:00:00.000000Z';
const fakeLine = (n) =>
  `${FAKE_TS}  INFO victory_gpu::host: frame ${n}, 30.0 fps, 33.3 ms/frame ` +
  '(0.4 preparing: copies 0.2 ms 0.3 MB, textures 0.0 ms 0.0 MB; 0.0 on the GPU; 0.0 copying back 0.0 MB), ' +
  '5000 draws/frame, dropped 0 without a host pipeline (0 being made) and 0 indirect';

function fakeRun(clock, dir, { shape = 'sample', failing = false } = {}) {
  const log = path.join(dir, FAKE_LOG);
  const stats = path.join(dir, 'stats.json');
  fs.writeFileSync(
    log,
    `${['harness: fake run', fakeLine(1), fakeLine(2), fakeLine(3), 'harness: settled at 5 s (waiting 0, 30.0 fps)'].join('\n')}\n`,
  );
  const started = clock.now();
  let seq = 0;
  const write = () => {
    const frame = 10 + Math.floor((clock.now() - started) / 50);
    seq += 1;
    // The interval shape: the fields the watchdog reads, by their names in fixtures/runtime-stats.mjs.
    const object =
      shape === 'interval'
        ? {
            seq, t_ms: clock.now() - started, unix_ms: clock.now(),
            interval: { first_frame: frame - 1, last_frame: frame, frames: 2, wall_ms: 66.6 },
            frame, fps: 30, frame_ms: 33.3, frame_ms_p50: 33.3, frame_ms_worst: 33.3, frame_ms_list: [33.3, 33.3],
          }
        : { frame, fps: 30, frame_ms: 33.3 };
    fs.writeFileSync(stats, `${JSON.stringify(object)}\n`);
  };
  write();
  const stopWrites = clock.every(50, write);
  clock.after(400, () => {
    stopWrites();
    if (failing) fs.writeFileSync(stats, '{"frame": ');
  });
  if (failing) {
    let n = 100;
    clock.every(200, () => fs.appendFileSync(log, `${fakeLine(n++)}\n`));
  }
  return { log, stats };
}

/// The process the watcher's end route may end: it does nothing but stay alive. Its pid goes where the stub reads it.
function standIn(dir) {
  const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { cwd: dir, stdio: 'ignore' });
  child.unref();
  fs.writeFileSync(path.join(dir, 'game.pid'), String(child.pid));
  return child;
}

function selectionStub(dir) {
  const psStub = path.join(dir, 'select.ps1');
  fs.writeFileSync(
    psStub,
    "Add-Content -Path (Join-Path $env:WATCHDOG_TEST_DIR 'selected.txt') -Value 'a selection'\n" +
      "Get-Content (Join-Path $env:WATCHDOG_TEST_DIR 'game.pid')\n",
  );
  return psStub;
}

function alive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err.code === 'EPERM';
  }
}

/// A wait on a real process (the stand-in's end, Windows letting go of a folder): a generous bound, for a slow,
/// loaded machine; what is asserted is that it happened, never how soon.
async function waitFor(condition, timeoutMs = 60_000) {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    if (condition()) return true;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  return condition();
}

/// Moves the virtual clock on in poll-sized steps until `condition` holds, at most `limitMs` of its time.
async function advanceUntil(clock, condition, limitMs = 30_000, stepMs = 100) {
  const until = clock.now() + limitMs;
  while (!condition()) {
    if (clock.now() >= until) return false;
    await clock.sleep(stepMs);
  }
  return true;
}

/// A promise's outcome as a box a condition can read.
function outcome(promise) {
  const box = { done: false, value: undefined };
  promise.then((value) => {
    box.done = true;
    box.value = value;
  });
  return box;
}

/// The test's folder, the selection's environment and the stand-in, cleaned up after: the stand-in is ended first,
/// because Windows cannot remove the folder its working directory is in.
function watchdogScene(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'watchdog-'));
  const previous = process.env.WATCHDOG_TEST_DIR;
  process.env.WATCHDOG_TEST_DIR = dir;
  const child = standIn(dir);
  t.after(async () => {
    if (previous === undefined) delete process.env.WATCHDOG_TEST_DIR;
    else process.env.WATCHDOG_TEST_DIR = previous;
    if (alive(child.pid)) {
      try {
        process.kill(child.pid);
      } catch {
        // Already gone.
      }
    }
    await waitFor(() => !alive(child.pid));
    // Windows can hold the process's working folder for a moment after the process is gone.
    for (let i = 0; ; i += 1) {
      try {
        fs.rmSync(dir, { recursive: true, force: true });
        return;
      } catch (err) {
        if (i >= 600) throw err;
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
    }
  });
  return { dir, child, psStub: selectionStub(dir), clock: virtualClock() };
}

test('the watcher ends a frozen fake run by the timer route, and the summary says so', async (t) => {
  const { dir, child, psStub, clock } = watchdogScene(t);
  const { log, stats } = fakeRun(clock, dir);
  const watching = outcome(watch(log, { stallSeconds: 1, pollMs: 100, statsFile: stats, psFile: psStub, clock }));
  assert.equal(await advanceUntil(clock, () => watching.done), true, 'the watcher ended within 30 s of its clock');
  assert.equal(watching.value, 0);
  assert.equal(await waitFor(() => !alive(child.pid)), true, 'the fake game was ended');
  assert.equal(fs.existsSync(path.join(dir, 'selected.txt')), true, 'the watcher selected the run through the strict selection script');

  const frozenFrame = JSON.parse(fs.readFileSync(stats, 'utf8')).frame;
  assert.equal(frozenFrame, 18, 'the counter stood at 400 ms of the fake clock: frame 10 + 400 / 50');
  const text = fs.readFileSync(log, 'utf8');
  const marker = text.split(/\r?\n/).find((line) => line.startsWith('harness: watchdog: the frame counter stopped'));
  assert.ok(marker, 'the stall line is in the run log');
  assert.match(marker, /^harness: watchdog: the frame counter stopped at frame \d+ for \d+ s; ending the run$/);
  assert.equal(marker.includes(`at frame ${frozenFrame} `), true, 'the line names the frozen frame');
  const seconds = Number(marker.match(/for (\d+) s; ending the run$/)[1]);
  assert.ok(seconds >= 1, `the line says how long it stood still (${seconds} s)`);

  const tailFile = tailPathFor(log);
  assert.equal(fs.existsSync(tailFile), true, 'the log tail is saved beside the log');
  const tail = fs.readFileSync(tailFile, 'utf8');
  assert.ok(tail.trimEnd().split(/\r?\n/).length <= TAIL_LINES);
  assert.equal(tail.includes(marker), true, 'the tail carries the watchdog line');

  const summary = spawnSync(process.execPath, [path.join(here, 'summarize-run.mjs'), log, '--json'], {
    encoding: 'utf8',
    env: isolatedEnv(),
  });
  assert.equal(summary.status, 0, summary.stderr);
  const parsed = JSON.parse(summary.stdout);
  assert.equal(parsed.ending.kind, 'watchdog');
  assert.equal(parsed.ending.frame, frozenFrame);
  assert.equal(parsed.ending.text, `ended: watchdog (frame ${frozenFrame}, ${seconds} s)`);
});

for (const shape of ['sample', 'interval'])
test(`warn mode reports a stall and its resume, never ends the fake run, and the summary counts it (${shape} shape)`, async (t) => {
  const { child, dir, psStub, clock } = watchdogScene(t);
  const { log, stats } = fakeRun(clock, dir, { shape });
  const watching = outcome(watch(log, { stallSeconds: 1, pollMs: 100, statsFile: stats, psFile: psStub, warn: true, clock }));

  const warnLine = () =>
    fs
      .readFileSync(log, 'utf8')
      .split(/\r?\n/)
      .find((line) => line.includes('NOT ending the run (warn mode)'));
  assert.equal(await advanceUntil(clock, () => Boolean(warnLine())), true, 'the warn stall line is in the run log');
  const marker = warnLine();
  assert.match(marker, /^harness: watchdog: the frame counter stopped at frame \d+ for \d+ s; NOT ending the run \(warn mode\)$/);
  const frozenFrame = Number(marker.match(/stopped at frame (\d+) /)[1]);
  const stallSeconds = Number(marker.match(/for (\d+) s;/)[1]);
  assert.ok(stallSeconds >= 1);
  assert.equal(alive(child.pid), true, 'warn mode left the fake game running');
  const textAfterStall = fs.readFileSync(log, 'utf8');
  assert.equal(textAfterStall.split(/\r?\n/).filter((line) => /; ending the run$/.test(line)).length, 0);
  assert.equal(textAfterStall.match(/NOT ending the run \(warn mode\)/g).length, 1, 'one stall line per stall');

  const tailFile = tailPathFor(log);
  assert.equal(fs.existsSync(tailFile), true, 'the tail is saved for the stall');
  assert.equal(fs.readFileSync(tailFile, 'utf8').includes(marker), true);

  // The counter moves again: the watcher says so and keeps watching. In the interval shape the next object holds the
  // whole gap (the runtime writes nothing while nothing presents): one long frame, then six ordinary ones.
  const resumedFrame = frozenFrame + 7;
  const last = JSON.parse(fs.readFileSync(stats, 'utf8'));
  const gapMs = clock.now() - (last.unix_ms ?? clock.now());
  fs.writeFileSync(
    stats,
    JSON.stringify(
      shape === 'interval'
        ? {
            ...last,
            seq: last.seq + 1,
            unix_ms: clock.now(),
            interval: { first_frame: frozenFrame + 1, last_frame: resumedFrame, frames: 7, wall_ms: gapMs + 6 * 33.3 },
            frame: resumedFrame,
            frame_ms: (gapMs + 6 * 33.3) / 7,
            frame_ms_list: [gapMs, 33.3, 33.3, 33.3, 33.3, 33.3, 33.3],
          }
        : { frame: resumedFrame, fps: 30, frame_ms: 33.3 },
    ) + '\n',
  );
  assert.equal(
    await advanceUntil(clock, () => fs.readFileSync(log, 'utf8').includes(`harness: watchdog: frames resumed at frame ${resumedFrame} after `)),
    true,
    'the resume line names the new frame',
  );
  assert.equal(alive(child.pid), true, 'the resume did not end the fake game');

  fs.appendFileSync(log, 'harness: victory exit=0\n');
  assert.equal(await advanceUntil(clock, () => watching.done), true);
  assert.equal(watching.value, 0);
  assert.equal(alive(child.pid), true, 'warn mode never called the ending route');

  const summary = spawnSync(process.execPath, [path.join(here, 'summarize-run.mjs'), log, '--json'], {
    encoding: 'utf8',
    env: isolatedEnv(),
  });
  assert.equal(summary.status, 0, summary.stderr);
  const parsed = JSON.parse(summary.stdout);
  assert.equal(parsed.watchdog.stalls, 1);
  assert.ok(parsed.watchdog.longestSeconds >= stallSeconds);
  assert.equal(parsed.ending.kind, 'clean_exit');

  const human = spawnSync(process.execPath, [path.join(here, 'summarize-run.mjs'), log], {
    encoding: 'utf8',
    env: isolatedEnv(),
  });
  assert.equal(human.status, 0, human.stderr);
  assert.match(human.stdout, new RegExp(`^watchdog: 1 stalls, longest \\d+ s$`, 'm'));
});

test('the watcher does not end a healthy run whose stats writes fail, warns once, and counts every failed read', async (t) => {
  const { child, dir, psStub, clock } = watchdogScene(t);
  const { log, stats } = fakeRun(clock, dir, { failing: true });
  // The ending mode, with a one-second window: today's rule ended this run about a second in.
  const watching = outcome(watch(log, { stallSeconds: 1, pollMs: 100, statsFile: stats, psFile: psStub, clock }));
  const warningLine = () =>
    fs
      .readFileSync(log, 'utf8')
      .split(/\r?\n/)
      .find((line) => line.startsWith('harness: watchdog: warning: '));
  assert.equal(await advanceUntil(clock, () => Boolean(warningLine())), true, 'the watcher warns that the stats file stopped');
  await clock.sleep(2500);
  assert.equal(alive(child.pid), true, 'a run that keeps presenting is not ended');
  const text = fs.readFileSync(log, 'utf8');
  assert.equal(text.includes('the frame counter stopped at frame'), false, 'no stall line');
  const warnings = text.split(/\r?\n/).filter((line) => line.startsWith('harness: watchdog: warning: '));
  assert.equal(warnings.length, 1, 'one warning for one episode');
  assert.match(warnings[0], /the log shows frames/);
  assert.match(warnings[0], /stats file: \d+ failed reads of \d+/);
  assert.match(warnings[0], /; NOT ending the run$/);

  fs.appendFileSync(log, 'harness: victory exit=0\n');
  assert.equal(await advanceUntil(clock, () => watching.done), true);
  assert.equal(watching.value, 0);
  const done = fs
    .readFileSync(log, 'utf8')
    .split(/\r?\n/)
    .find((line) => line.startsWith('harness: watchdog: done: '));
  assert.ok(done, 'the watcher closes with its tally');
  const failed = Number(done.match(/stats file: (\d+) failed reads of (\d+)/)[1]);
  assert.ok(failed >= 25, `one failed read a poll for over 2.5 s of its clock, all counted (${failed})`);
});

test('the watcher uses the hold timer’s selection and end route, not a new one', () => {
  const watcher = fs.readFileSync(path.join(here, 'watchdog.mjs'), 'utf8');
  assert.match(watcher, /from '\.\/settled-watch\.mjs'/);
  assert.match(watcher, /\bselectionPids,/);
  assert.match(watcher, /\bendPid,/);
  assert.match(watcher, /\bisAlive,/);
  assert.doesNotMatch(watcher, /spawnSync\(\s*'taskkill'/);
  assert.doesNotMatch(watcher, /Stop-Process|Get-CimInstance|reg\s+add/);

  const settled = fs.readFileSync(path.join(here, 'settled-watch.mjs'), 'utf8');
  assert.match(settled, /spawnSync\(\s*'taskkill', \['\/F', '\/PID'/);
  assert.match(settled, /'-File', psFile, '-Pids', '-Strict'/);
});

// A game that ends by itself leaves its process id free, and Windows gives it out again within seconds. The stand-in
// here plays the program that got the id: alive, but no longer in the run's selection. The watcher must end only what
// a fresh selection returns at the moment of the kill (end-run.mjs), never the id it saw earlier.
test('a stall after the game has left the selection ends nothing, even while its old process id is alive', async (t) => {
  const { dir, child, psStub, clock } = watchdogScene(t);
  const { log, stats } = fakeRun(clock, dir);
  const watching = outcome(watch(log, { stallSeconds: 1, pollMs: 100, statsFile: stats, psFile: psStub, clock }));
  assert.equal(await advanceUntil(clock, () => fs.existsSync(path.join(dir, 'selected.txt'))), true, 'the watcher saw the game');
  fs.writeFileSync(path.join(dir, 'game.pid'), ''); // the game is gone from the selection; its id lives on
  assert.equal(await advanceUntil(clock, () => watching.done), true, 'the watcher ended within 30 s of its clock');
  assert.equal(watching.value, 0);
  assert.equal(alive(child.pid), true, 'the process holding the old id was not ended');
  assert.ok(!fs.readFileSync(log, 'utf8').includes('ending the run'), 'no run was ended');
});
