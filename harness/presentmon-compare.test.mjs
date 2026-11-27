// presentmon-compare.mjs on synthetic PresentMon CSVs and run logs made here (no real capture or log is copied).
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { frameLineOver, segmentFigures, segmentLine as runtimeSegmentLine, statsObjectOver } from './live-fake-runtime.mjs';
import { compareRun, frameStats, readFrameTimes } from './presentmon-compare.mjs';
import { oldSegmentLine } from './fixtures/old-shape.mjs';
import { isolatedEnv } from './test-isolation.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const ZERO = Date.parse('2026-11-26T10:00:00Z');
const stamp = (seconds) => new Date(ZERO + seconds * 1000).toISOString().replace('Z', '000Z');

// The game's process id as the run records it (run-world.sh appends the priority line to the log).
const PID = 100;
const PRIORITY = `priority: above normal for pid ${PID} (this run's game)`;

// One runtime frame line every 5 s from the capture's zero, `fps` frames a second, with the watcher's start line
// after the first (the line that triggered PresentMon).
// Corrected (was: no process id in the log, so PresentMon's rows of every process were pooled).
function runLog({ fps = 30, seconds = 60, segment = null } = {}) {
  const lines = ['harness build: abc1234 profile=diag console=series-x', PRIORITY];
  let frame = 1000;
  for (let s = 0; s <= seconds; s += 5) {
    lines.push(
      `${stamp(s)}  INFO victory_gpu::host: frame ${frame}, ${fps} fps, ${(1000 / fps).toFixed(1)} ms/frame (x), ` +
        '5000 draws/frame, dropped 0 without a host pipeline (0 being made) and 0 indirect',
    );
    if (s === 0) lines.push('presentmon: started at 5000 draws + 0 dropped per frame (threshold 1000), writing presentmon.csv');
    frame += fps * 5;
  }
  if (segment) lines.push(segment);
  lines.push('harness: victory exit=0');
  return lines;
}

// PresentMon 2.6's header (TimeInMs, MsBetweenPresents): one row per frame, `msAt(t, i)` frame i's time at t seconds.
function csv26(msAt, seconds = 60, { offset = 0 } = {}) {
  const rows = ['Application,ProcessID,TimeInMs,MsBetweenPresents,MsGPUBusy'];
  for (let t = offset, i = 0; t < seconds; i += 1) {
    const ms = msAt(t, i);
    rows.push(`victory.exe,100,${(t * 1000).toFixed(3)},${ms.toFixed(3)},5`);
    t += ms / 1000;
  }
  return rows.join('\r\n') + '\r\n';
}

test('frameStats: mean, median and nearest-rank p95 of a known set', () => {
  const values = Array.from({ length: 20 }, (_, i) => i + 1); // 1..20
  assert.deepEqual(frameStats(values), { count: 20, mean: 10.5, median: 10.5, p95: 19 });
  assert.deepEqual(frameStats([]), { count: 0, mean: null, median: null, p95: null });
});

// Corrected (was: rows carried no process id, and the result no pidColumn).
test('readFrameTimes reads both header spellings, their time units and the process id', () => {
  const v26 = readFrameTimes('﻿Application,ProcessID,TimeInMs,MsBetweenPresents\r\nvictory.exe,7,1500,33.3\r\nvictory.exe,7,1533,NA\r\n');
  assert.deepEqual(v26, { frameColumn: 'msbetweenpresents', timeColumn: 'timeinms', pidColumn: true, rows: [{ seconds: 1.5, ms: 33.3, pid: 7 }] });
  const v2 = readFrameTimes('Application,CPUStartTime,FrameTime\nvictory.exe,2000,16.7\n');
  assert.deepEqual(v2, { frameColumn: 'frametime', timeColumn: 'cpustarttime', pidColumn: false, rows: [{ seconds: 2, ms: 16.7, pid: null }] });
  const v1 = readFrameTimes('Application,TimeInSeconds,msBetweenPresents\nvictory.exe,2.5,20\n');
  assert.deepEqual(v1, { frameColumn: 'msbetweenpresents', timeColumn: 'timeinseconds', pidColumn: false, rows: [{ seconds: 2.5, ms: 20, pid: null }] });
  assert.equal(readFrameTimes('Application,TimeInMs\nvictory.exe,1\n').frameColumn, null);
});

// Corrected (was "(mean)" in the verdict: the frame count is gated too).
test('a runtime agreeing with PresentMon passes; the median and p95 are not checked without per-frame figures', () => {
  const result = compareRun(runLog({ fps: 30 }), csv26(() => 1000 / 30));
  assert.equal(result.verdict, 'PASS', result.lines.join('\n'));
  assert.equal(result.code, 0);
  assert.match(result.lines.join('\n'), /mean\s+runtime 33\.3 ms \(frames 1000\.\.2800 by its counter\) against PresentMon 33\.3 ms: [+-]0\.0% agrees/);
  assert.match(result.lines.join('\n'), /median PresentMon 33\.3 ms; runtime n\/a .*not checked/);
  assert.match(result.lines.at(-1), /^PASS: the runtime agrees with PresentMon within 5% \(mean, frames\)$/);
});

test('a runtime differing from PresentMon past the tolerance fails, and names the figure', () => {
  // PresentMon sees 40 ms frames where the runtime's counter says 33.3 ms: the runtime is 16.7% under it.
  const result = compareRun(runLog({ fps: 30 }), csv26(() => 40));
  assert.equal(result.verdict, 'FAIL');
  assert.equal(result.code, 1);
  // Corrected (was "(mean)": the frame count is gated too, and 40 ms frames are fewer frames than the counter's).
  assert.match(result.lines.at(-1), /^FAIL: the runtime differs from PresentMon by more than 5% \(mean, frames\)$/);
  // A wider tolerance takes it.
  assert.equal(compareRun(runLog({ fps: 30 }), csv26(() => 40), { tolerancePct: 25 }).verdict, 'PASS');
});

test('the window keeps PresentMon frames outside it out of every figure', () => {
  // 50 ms frames for the first 30 s, 25 ms after; the runtime's counter says 30 fps throughout, so only the
  // window's own frames decide: 30-60 s is all 25 ms.
  const result = compareRun(runLog({ fps: 30 }), csv26((t) => (t < 30 ? 50 : 25)), { fromS: 30, toS: 60 });
  const text = result.lines.join('\n');
  assert.match(text, /window 30\.0-60\.0 s of the capture/);
  assert.match(text, /mean\s+runtime 33\.3 ms .* against PresentMon 25\.0 ms/);
});

test('a live segment gates the median and p95 against its per-frame figures, over its own frames', () => {
  // Frames 1000..2800 at 30 fps span 0-60 s; segment 2 covers frames 1300..2200, 10-40 s.
  // The pre-rebuild line (fixtures/old-shape.mjs): every present counted.
  const segment = oldSegmentLine(2, 1300, 2200, 33.3).replace('p95 35.3', 'p95 40.0');
  // One frame in ten at 40 ms: per frame, the median is 33 and the p95 40.
  const csv = csv26((t, i) => (i % 10 === 0 ? 40 : 33));
  const result = compareRun(runLog({ fps: 30, segment }), csv, { segment: 2 });
  const text = result.lines.join('\n');
  assert.match(text, /window 10\.0-40\.0 s of the capture/);
  assert.match(text, /p95\s+runtime 40\.0 ms \(live segment 2\) against PresentMon 40\.0 ms: \+0\.0% agrees/);
  assert.equal(result.verdict, 'PASS', text);
  // Corrected (was "(mean, median, p95)": the segment's frame count is gated too).
  assert.match(result.lines.at(-1), /\(mean, median, p95, frames\)$/);
  // A segment the log does not have cannot be checked.
  assert.equal(compareRun(runLog({ segment }), csv, { segment: 9 }).verdict, 'CANNOT CHECK');
});

// Corrected (was: too few frames was CANNOT CHECK, exit 2; it is COULD NOT COMPARE, exit 3, since the inputs are there
// and only their overlap is short).
test('no CSV, no frame time column or no start line: CANNOT CHECK; too few frames: COULD NOT COMPARE; never a pass', () => {
  const log = runLog();
  for (const [csv, why] of [
    [null, /no PresentMon CSV/],
    ['Application,TimeInMs\nvictory.exe,1\n', /no frame time column/],
  ]) {
    const result = compareRun(log, csv);
    assert.equal(result.verdict, 'CANNOT CHECK');
    assert.equal(result.code, 2);
    assert.match(result.lines.at(-1), why);
  }
  const few = compareRun(log, `Application,ProcessID,TimeInMs,MsBetweenPresents\nvictory.exe,${PID},1,33\nvictory.exe,${PID},34,33\n`);
  assert.equal(few.verdict, 'COULD NOT COMPARE', few.lines.join('\n'));
  assert.equal(few.code, 3);
  const noStart = log.filter((line) => !line.startsWith('presentmon: started'));
  assert.match(compareRun(noStart, csv26(() => 33.3)).lines.at(-1), /no "presentmon: started" line/);
});

test('the command line reads the CSV beside the log by default and exits by its verdict', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'presentmon-compare-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const capture = path.join(dir, 'capture');
  fs.mkdirSync(capture);
  const log = path.join(dir, 'run.log');
  fs.writeFileSync(log, [`capture: ${capture}`, ...runLog({ fps: 30 })].join('\n') + '\n');
  fs.writeFileSync(path.join(capture, 'presentmon.csv'), csv26(() => 1000 / 30));
  const script = path.join(here, 'presentmon-compare.mjs');
  const pass = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(pass.status, 0, pass.stderr);
  assert.match(pass.stdout, /^PASS: /m);
  assert.equal(pass.stdout.includes(dir), false, 'no folder path is printed');
  fs.writeFileSync(path.join(capture, 'presentmon.csv'), csv26(() => 40));
  assert.equal(spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() }).status, 1);
  fs.rmSync(path.join(capture, 'presentmon.csv'));
  const none = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(none.status, 2);
  assert.match(none.stdout, /^CANNOT CHECK: no PresentMon CSV/m);
  assert.equal(spawnSync(process.execPath, [script], { encoding: 'utf8', env: isolatedEnv() }).status, 2);
});

test('the 28-column header PresentMon 2.6.0 writes: TimeInMs is the clock, MsBetweenPresents the frame time', () => {
  const header =
    '\uFEFFApplication,ProcessID,SwapChainAddress,PresentRuntime,SyncInterval,PresentFlags,AllowsTearing,PresentMode,' +
    'TimeInMs,MsBetweenSimulationStart,MsBetweenPresents,MsBetweenDisplayChange,MsInPresentAPI,MsRenderPresentLatency,' +
    'MsUntilDisplayed,CPUStartTimeInMs,MsBetweenAppStart,MsCPUBusy,MsCPUWait,MsGPULatency,MsGPUTime,MsGPUBusy,MsGPUWait,' +
    'MsAnimationError,AnimationTime,MsFlipDelay,MsAllInputToPhotonLatency,MsClickToPhotonLatency';
  // CPUStartTimeInMs is deliberately far off: the reader must take TimeInMs.
  const rows = [];
  for (let i = 0, t = 0; t < 60_000; i += 1, t += 1000 / 30) {
    rows.push(`victory.exe,100,0x1,DXGI,0,0,1,Hardware: Independent Flip,${t.toFixed(3)},NA,${(1000 / 30).toFixed(3)},33.3,0.5,0.6,0.7,${(t + 900000).toFixed(3)},33.3,30,3,0.2,25,24,9,NA,${t.toFixed(3)},NA,NA,NA`);
  }
  const parsed = readFrameTimes(`${header}\r\n${rows.join('\r\n')}\r\n`);
  assert.equal(parsed.timeColumn, 'timeinms');
  assert.equal(parsed.frameColumn, 'msbetweenpresents');
  const result = compareRun(runLog({ fps: 30 }), `${header}\r\n${rows.join('\r\n')}\r\n`);
  assert.equal(result.verdict, 'PASS', result.lines.join('\n'));
});

test('the difference is the runtime against PresentMon, so PresentMon is the reference the percentage is taken of', () => {
  const result = compareRun(runLog({ fps: 30 }), csv26(() => 40));
  assert.match(result.lines.join('\n'), /mean\s+runtime 33\.3 ms .* against PresentMon 40\.0 ms: -16\.7% DISAGREES/);
});

// A made-up title on a virtual clock, generated here the way the check of 2026-11-26 generated its probe (no real log,
// frame or CSV): frames whose times `frameMs(t)` gives; frame lines in the window shape every 5 s; the watcher's start
// line after the first frame line at or past `triggerAt`; one stats object a second carrying every frame's time; and
// segment lines, a change at each time in `cuts` (`open`: the run ends by the timer, so the last segment has only its
// open lines). `gain` scales the frame counter (a wrong meter); `listGain` the stats objects' frame times. Every line
// and object is in the runtime's shape, written by the fake runtime's writers (live-fake-runtime.test.mjs shows they
// give the fixtures copied from the runtime's own tests).
const T0 = Date.parse('2026-11-26T12:00:00Z');
const at = (t) => new Date(T0 + t).toISOString();
function title({ seconds, frameMs, triggerAt = 10_000, gain = () => 1, listGain = 1, cuts = [], open = false, record = true }) {
  const lines = ['harness build: abc1234 profile=diag console=series-x'];
  if (record) lines.push(PRIORITY);
  const presents = [];
  const stats = [];
  const sum = (xs) => xs.reduce((a, b) => a + b, 0);
  const near = (xs, p) => {
    const sorted = [...xs].sort((a, b) => a - b);
    return sorted.length ? sorted[Math.max(1, Math.ceil((sorted.length * p) / 100)) - 1] : 0;
  };
  let t = 0;
  let counter = 0;
  let zero = null;
  let line = { at: 0, last: 0, times: [] };
  let object = { at: 0, last: 0, times: [] };
  let segment = { number: 1, first: 1, last: 1, times: [], begun: 'start' };
  const left = [...cuts];
  const segmentLine = (s, end) =>
    `${at(t)}  INFO victory_gpu::live: ${runtimeSegmentLine({
      k: s.number,
      first: s.first,
      last: s.last,
      figures: segmentFigures(s.times.map((ms) => ({ ms })), s.begun),
      begun: s.begun,
      ended: end === 'open' ? null : end,
      inForce: 'VICTORY_FETCH_QUEUE=on',
    })}`;
  while (t < seconds * 1000) {
    // A change at a frame boundary: it closes the segment at the last frame presented (its marker says that frame),
    // and the next segment begins with the next frame, its first three left out as transition (live.rs `poll`).
    if (left.length && t >= left[0]) {
      left.shift();
      lines.push(segmentLine(segment, 'change'));
      lines.push(`${at(t)}  INFO victory_gpu::live: live: VICTORY_FETCH_QUEUE off -> on at frame ${segment.last}`);
      segment = { number: segment.number + 1, first: null, last: null, times: [], begun: 'change' };
    }
    const dt = frameMs(t);
    t += dt;
    counter += gain(t);
    const n = Math.round(counter);
    presents.push({ t, dt, n });
    segment.first ??= n;
    segment.times.push(dt);
    segment.last = n;
    line.times.push(dt);
    object.times.push(dt);
    if (t - object.at >= 1000) {
      const list = object.times.map((x) => x * listGain);
      const written = statsObjectOver({
        seq: stats.length + 1,
        tMs: t,
        unixMs: T0 + t,
        firstFrame: object.last + 1,
        times: list,
        live: { VICTORY_FETCH_QUEUE: 'on' },
        closed: segment.number - 1,
        open: null,
      });
      // The frame counter is the test's own (a wrong meter scales it): the object's last frame is its count.
      written.interval.last_frame = n;
      written.frame = n;
      stats.push(written);
      object = { at: t, last: n, times: [] };
    }
    if (t - line.at >= 5000) {
      const frames = n - line.last;
      const wall = sum(line.times);
      lines.push(`${at(t)}  INFO victory_gpu::host: ${frameLineOver({ first: line.last + 1, last: n, wallMs: wall, p50: near(line.times, 50), p95: near(line.times, 95) })}`);
      if (zero === null && t >= triggerAt) {
        zero = t;
        lines.push('presentmon: started at 1500 draws + 0 dropped per frame (threshold 1000), writing presentmon.csv');
      }
      if (open) lines.push(segmentLine(segment, 'open'));
      line = { at: t, last: n, times: [] };
    }
  }
  if (!open) lines.push(segmentLine(segment, 'end'));
  lines.push('harness: victory exit=0');
  return { lines, presents, zero, stats };
}

// PresentMon's CSV for the presents in [zero + lateMs, zero + lateMs + seconds): its clock starts when it does. `lose`
// drops an event (the next row then carries the time since the last one it saw); `others` adds another process's rows.
function capture(run, { seconds = 60, lateMs = 0, lose = () => false, others = [] } = {}) {
  const begin = run.zero + lateMs;
  const rows = [];
  let carried = 0;
  let seen = 0;
  for (const p of run.presents) {
    if (p.t < begin || p.t >= begin + seconds * 1000) continue;
    seen += 1;
    if (lose(seen)) {
      carried += p.dt;
      continue;
    }
    rows.push({ at: p.t - begin, ms: p.dt + carried, pid: PID });
    carried = 0;
  }
  for (const other of others) for (let t = 0; t < seconds * 1000; t += other.ms) rows.push({ at: t, ms: other.ms, pid: other.pid });
  rows.sort((a, b) => a.at - b.at);
  const body = rows.map((r) => `victory.exe,${r.pid},${r.at.toFixed(4)},${r.ms.toFixed(4)},5`).join('\r\n');
  return `Application,ProcessID,TimeInMs,MsBetweenPresents,MsGPUBusy\r\n${body}\r\n`;
}

test("the window ends at the capture's last row: a right meter whose scene slows after the capture passes", () => {
  // The check's default-window case: a 60 s capture, then 120 s more at 400 ms. The runtime's meter is exactly right.
  const run = title({ seconds: 190, frameMs: (t) => (t < 70_000 ? 250 : 400) });
  const result = compareRun(run.lines, capture(run));
  const text = result.lines.join('\n');
  assert.equal(result.verdict, 'PASS', text);
  assert.match(text, /window 0\.0-5\d\.\d s of the capture/);
  assert.match(text, /mean\s+runtime 250\.0 ms/);
});

test('a meter wrong only after the capture is judged on the capture alone, and both frame counts are printed', () => {
  // The counter runs 20% fast after 70 s, beyond the capture: nothing PresentMon saw can judge that part, so the window
  // must say it ends inside the capture (it said 0-180 s), with PresentMon's frames and the runtime's beside each other.
  const run = title({ seconds: 190, frameMs: (t) => (t < 70_000 ? 250 : 300), gain: (t) => (t < 70_000 ? 1 : 1.2) });
  const result = compareRun(run.lines, capture(run));
  const text = result.lines.join('\n');
  assert.match(text, /window 0\.0-5\d\.\d s of the capture/);
  assert.match(text, /PresentMon: \d+ frames/);
  assert.match(text, /runtime: \d+ frames by its frame counter/);
  assert.match(text, /frames\s+runtime \d+ against PresentMon \d+: [+-]\d+\.\d% agrees/);
});

test("PresentMon's rows are kept for the game's process alone, by the id the run recorded", () => {
  // Another process presents at 60 fps in the same file: pooled, its rows pulled the median to 16.7 ms.
  const run = title({ seconds: 70, frameMs: () => 250 });
  const result = compareRun(run.lines, capture(run, { others: [{ pid: 4242, ms: 16.667 }] }));
  const text = result.lines.join('\n');
  assert.equal(result.verdict, 'PASS', text);
  assert.match(text, new RegExp(`of process ${PID} \\(its id from the priority line; \\d+ rows of other processes left out\\)`));
});

test('a run that recorded no process id is refused, never pooled; --pid names one', () => {
  const run = title({ seconds: 70, frameMs: () => 250, record: false });
  const csv = capture(run);
  const refused = compareRun(run.lines, csv);
  assert.equal(refused.verdict, 'CANNOT CHECK', refused.lines.join('\n'));
  assert.equal(refused.code, 2);
  assert.match(refused.lines.at(-1), /recorded no process id for the game/);
  assert.equal(compareRun(run.lines, csv, { pid: PID }).verdict, 'PASS');
  // A recorded id with no rows in the CSV is refused too, not compared on another process's rows.
  const elsewhere = compareRun(run.lines, csv, { pid: 777 });
  assert.equal(elsewhere.verdict, 'CANNOT CHECK');
  assert.match(elsewhere.lines.at(-1), /no PresentMon row belongs to process 777/);
});

test("lost presents show as fewer rows than the runtime's counter, and the frame count is gated", () => {
  const run = title({ seconds: 70, frameMs: () => 250 });
  const lost = compareRun(run.lines, capture(run, { lose: (i) => i % 12 === 11 }));
  assert.equal(lost.verdict, 'FAIL', lost.lines.join('\n'));
  assert.match(lost.lines.at(-1), /\(.*frames\)$/);
  assert.match(lost.lines.join('\n'), /frames\s+runtime \d+ against PresentMon \d+: \+\d+\.\d% DISAGREES/);
  assert.equal(compareRun(run.lines, capture(run, { lose: (i) => i % 100 === 99 })).verdict, 'PASS');
});

test('a window outside the capture, or an overlap under the minimum, is COULD NOT COMPARE with its own exit code', () => {
  const run = title({ seconds: 190, frameMs: () => 250 });
  const csv = capture(run);
  for (const window of [{ fromS: 90, toS: 150 }, { fromS: 0, toS: 6 }]) {
    const result = compareRun(run.lines, csv, window);
    assert.equal(result.verdict, 'COULD NOT COMPARE', result.lines.join('\n'));
    assert.equal(result.code, 3);
  }
});

test("with the run's stats history the mean, median and p95 come from every frame in the same window", () => {
  // Every tenth frame 500 ms, the rest 200 ms: per frame the median is 200 and the p95 500.
  const frameMs = () => {
    frameMs.i = (frameMs.i ?? 0) + 1;
    return frameMs.i % 10 === 0 ? 500 : 200;
  };
  const run = title({ seconds: 80, frameMs });
  const result = compareRun(run.lines, capture(run), { statsObjects: run.stats });
  const text = result.lines.join('\n');
  assert.equal(result.verdict, 'PASS', text);
  assert.match(text, /median\s+runtime 200\.0 ms \(the stats objects' frame lists[^)]*\) against PresentMon 200\.0 ms: \+0\.0% agrees/);
  assert.match(text, /p95\s+runtime 500\.0 ms .* against PresentMon 500\.0 ms: \+0\.0% agrees/);
  assert.match(result.lines.at(-1), /\(mean, median, p95, frames\)$/);
  // Frame times 20% short in the stats objects: a wrong meter, caught on every figure taken from them.
  frameMs.i = 0;
  const wrong = title({ seconds: 80, frameMs, listGain: 0.8 });
  const failed = compareRun(wrong.lines, capture(wrong), { statsObjects: wrong.stats });
  assert.equal(failed.verdict, 'FAIL', failed.lines.join('\n'));
  assert.match(failed.lines.at(-1), /\(mean, median, p95\)$/);
});

// The change at 25 s makes its first three frames slow (2 s each): the counted segment leaves them out, and so must
// PresentMon's window; with them in, PresentMon's mean is about 9% higher.
function changedRun(options = {}) {
  let after = 0;
  const frameMs = (t) => (t < 25_000 ? 250 : ++after <= 3 ? 2000 : 200);
  return title({ seconds: 65, frameMs, cuts: [25_000], ...options });
}

test('--segment K reads the counted segment line: its counted frames, the transition left out of both sides', () => {
  const run = changedRun();
  const result = compareRun(run.lines, capture(run), { segment: 2, statsObjects: run.stats });
  const text = result.lines.join('\n');
  assert.equal(result.verdict, 'PASS', text);
  assert.match(text, /mean\s+runtime \d+\.\d ms \(live segment 2\)/);
  assert.match(text, /p95\s+runtime 200\.0 ms \(live segment 2\) against PresentMon 200\.0 ms/);
  assert.match(text, /live segment 2: the runtime's shape, \d+ counted, 3 transition frames left out, no light or long frame/);
});

test('--segment K takes the open segment of a run the timer ended, which has no closing line', () => {
  const run = changedRun({ open: true });
  assert.equal(run.lines.some((line) => /live segment 2: /.test(line)), false, 'no closing line');
  const result = compareRun(run.lines, capture(run), { segment: 2, statsObjects: run.stats });
  const text = result.lines.join('\n');
  assert.equal(result.verdict, 'PASS', text);
  assert.match(text, /live segment 2: the runtime's shape, open \(the segment under way at the log's end\)/);
});

test('a segment the capture only partly covers is COULD NOT COMPARE, never judged on the part PresentMon saw', () => {
  let after = 0;
  const run = title({ seconds: 130, frameMs: (t) => (t < 50_000 ? 250 : ++after <= 3 ? 2000 : 300), cuts: [50_000] });
  const result = compareRun(run.lines, capture(run), { segment: 2, statsObjects: run.stats });
  assert.equal(result.verdict, 'COULD NOT COMPARE', result.lines.join('\n'));
  assert.equal(result.code, 3);
  assert.match(result.lines.at(-1), /does not cover live segment 2/);
});

test('--segment K is refused when the segment left light or long frames out: PresentMon would hold frames the runtime does not', () => {
  // The run's first segment holds the load: frames over 1000 ms, left out of its figures as long.
  const run = title({ seconds: 65, frameMs: (t) => (t < 12_000 ? 1500 : 250), triggerAt: 1000, cuts: [40_000] });
  const result = compareRun(run.lines, capture(run, { seconds: 64 }), { segment: 1, statsObjects: run.stats });
  assert.equal(result.verdict, 'COULD NOT COMPARE', result.lines.join('\n'));
  assert.match(result.lines.at(-1), /live segment 1 left 0 light and \d+ long frames out of its figures/);
});

// The capture now runs the whole game (presentmon.mjs: zero seconds is until the game exits), load-in included: a
// minute at about 1.4 frames a second, then the settled scene at 30.
function wholeRun() {
  const frameMs = (t) => (t < 60_000 ? 714 : 1000 / 30);
  const run = title({ seconds: 200, frameMs, triggerAt: 1000 });
  return { run, csv: capture(run, { seconds: 200 }) };
}

test('a window on the wall clock is judged on its own rows: a minute of load-in in the capture changes nothing', () => {
  const { run, csv } = wholeRun();
  const from = T0 + 100_000;
  const to = T0 + 160_000;
  const result = compareRun(run.lines, csv, { statsObjects: run.stats, fromUnixMs: from, toUnixMs: to });
  const text = result.lines.join('\n');
  assert.equal(result.verdict, 'PASS', text);
  // Both sides on the same 60 s, with their frame counts printed, and only settled frames in either.
  const inside = run.presents.filter((p) => T0 + p.t > from && T0 + p.t <= to);
  assert.match(text, new RegExp(`unix ms ${from}\\.\\.${to}`));
  assert.match(text, new RegExp(`PresentMon: ${inside.length} frames`));
  assert.match(text, /median\s+runtime 33\.3 ms .* against PresentMon 33\.3 ms/);
  assert.match(text, /p95\s+runtime 33\.3 ms .* against PresentMon 33\.3 ms/);
  assert.match(text, /the capture's zero: 2026-11-26T12:00:0\d\.\d+Z \(unix ms \d+\)/);
  // The whole capture as the window takes the load-in in: its mean is the load-in's as much as the scene's.
  const whole = compareRun(run.lines, csv, { statsObjects: run.stats });
  const mean = Number(/mean\s+runtime ([\d.]+) ms/.exec(whole.lines.join('\n'))[1]);
  assert.ok(mean > 40, `the whole capture's mean ${mean} ms holds the load-in`);
});

test('a window by two stats objects does the arithmetic itself: from the present before the first to the second', () => {
  const { run, csv } = wholeRun();
  const first = run.stats.find((o) => o.unix_ms >= T0 + 100_000);
  const last = run.stats.find((o) => o.unix_ms >= T0 + 160_000);
  const bySeq = compareRun(run.lines, csv, { statsObjects: run.stats, fromSeq: first.seq, toSeq: last.seq });
  const text = bySeq.lines.join('\n');
  assert.equal(bySeq.verdict, 'PASS', text);
  const from = first.unix_ms - first.interval.wall_ms;
  assert.match(text, new RegExp(`asked: the window ${from}\\.\\.${last.unix_ms} unix ms \\(stats objects seq ${first.seq}\\.\\.${last.seq}`));
  // The frames of exactly those objects, on the runtime's side.
  const frames = run.stats.filter((o) => o.seq >= first.seq && o.seq <= last.seq).reduce((sum, o) => sum + o.interval.frames, 0);
  assert.match(text, new RegExp(`runtime ${frames} against PresentMon \\d+`));
  assert.equal(compareRun(run.lines, csv, { statsObjects: run.stats, fromSeq: 99_999, toSeq: 100_000 }).verdict, 'CANNOT CHECK');
  assert.equal(compareRun(run.lines, csv, { fromSeq: first.seq, toSeq: last.seq }).verdict, 'CANNOT CHECK', 'no stats history');
});

test('a window on the wall clock that the capture or the runtime does not cover is refused, never clipped to the part seen', () => {
  const { run } = wholeRun();
  const short = capture(run, { seconds: 120 });
  const result = compareRun(run.lines, short, { statsObjects: run.stats, fromUnixMs: T0 + 100_000, toUnixMs: T0 + 160_000 });
  assert.equal(result.verdict, 'COULD NOT COMPARE', result.lines.join('\n'));
  assert.equal(result.code, 3);
  assert.match(result.lines.at(-1), /the capture \([\d.]+-[\d.]+ s\) does not cover the window/);
  const late = compareRun(run.lines, capture(run, { seconds: 200 }), { statsObjects: run.stats, fromUnixMs: T0 + 150_000, toUnixMs: T0 + 260_000 });
  assert.equal(late.verdict, 'COULD NOT COMPARE');
  assert.equal(compareRun(run.lines, short, { statsObjects: run.stats, fromUnixMs: T0 + 1 }).verdict, 'CANNOT CHECK', 'one end alone');
  assert.equal(compareRun(run.lines, short, { statsObjects: run.stats, fromUnixMs: T0, toUnixMs: T0 + 1, fromS: 3 }).verdict, 'CANNOT CHECK', 'two windows');
});

// The two rules the check of 2026-11-26 found no test would catch if removed (its mutation section).
test('an overlap under MIN_OVERLAP_S is COULD NOT COMPARE even with frames enough on both sides', () => {
  // 60 fps for 9 s inside the capture: 540 frames a side, well over MIN_FRAMES, but under 10 s of overlap.
  const run = title({ seconds: 40, frameMs: () => 1000 / 60, triggerAt: 1000 });
  const csv = capture(run, { seconds: 39 });
  const from = run.zero + 5_000;
  const result = compareRun(run.lines, csv, { statsObjects: run.stats, fromUnixMs: T0 + from, toUnixMs: T0 + from + 9_000 });
  assert.equal(result.verdict, 'COULD NOT COMPARE', result.lines.join('\n'));
  assert.match(result.lines.at(-1), /overlap 9\.0 s, under 10 s/);
  const longer = compareRun(run.lines, csv, { statsObjects: run.stats, fromUnixMs: T0 + from, toUnixMs: T0 + from + 11_000 });
  assert.equal(longer.verdict, 'PASS', longer.lines.join('\n'));
});

test("a repeated stats object (the same seq read twice) is one object: its frames are not counted twice", () => {
  const run = title({ seconds: 80, frameMs: () => 200 });
  const csv = capture(run);
  const once = compareRun(run.lines, csv, { statsObjects: run.stats });
  // The recorder read every object twice, a file time moving with nothing new written.
  const twice = compareRun(run.lines, csv, { statsObjects: run.stats.flatMap((object) => [object, structuredClone(object)]) });
  assert.equal(once.verdict, 'PASS', once.lines.join('\n'));
  assert.deepEqual(twice.lines, once.lines);
});
