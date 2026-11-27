import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  CHANGE_LINE,
  MARK_LINE,
  SEGMENT_BEGUN_AT_MARK_LINE,
  SEGMENT_CLOSED_LINE,
  SEGMENT_ENDED_BY_MARK_LINE,
  SEGMENT_FIRST_LINE,
  SEGMENT_OPEN_LINE,
  SEGMENT_RUN_END_LINE,
  SEGMENT_TIMER_OPEN_LINE,
} from './fixtures/runtime-lines.mjs';
import { formatReport, lastOpenSegment, parseLiveLine, parseLiveLog, SHORT_FRAMES } from './live-report.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'live-report.mjs');

function tempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'live-report-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

const DIM = '\x1b[2m';
const RESET = '\x1b[0m';
const GREEN = '\x1b[32m';
const prefix = `${DIM}2026-11-26T10:00:00.000000Z${RESET} ${GREEN} INFO${RESET} ${DIM}victory_gpu::live${RESET}: `;

// Pre-rebuild lines (the older shape, every present counted), kept for reading old logs, with colour codes and CRLF.
// Segment 2 is 50 frames (short); its in-force lists are empty.
// The marker that cuts segment 3 has an empty old value, as a skip list that starts empty does.
const LOG = [
  `${prefix}live segment 1: frames 100..400, mean 16.7, p50 16.1, p95 21.3, skipped 0 draws and 0 dispatches, in force: VICTORY_KEEP_TEXTURES=on VICTORY_TEXTURE_LOW=70 VICTORY_TEXTURE_HIGH=85 VICTORY_LATE_WAITS=off LIVE_SKIP_DISPATCH= LIVE_SKIP_DRAW=`,
  `${prefix}live: VICTORY_TEXTURE_LOW 70 -> 60 at frame 401`,
  `${prefix}live segment 2: frames 401..450, mean 20.0, p50 19.2, p95 25.9, skipped 0 draws and 0 dispatches, in force: VICTORY_KEEP_TEXTURES=on VICTORY_TEXTURE_LOW=60 VICTORY_TEXTURE_HIGH=85 VICTORY_LATE_WAITS=off LIVE_SKIP_DISPATCH= LIVE_SKIP_DRAW=`,
  `${prefix}live: LIVE_SKIP_DRAW  -> 1234,5678 at frame 451`,
  `${prefix}live: VICTORY_KEEP_TEXTURES on -> off at frame 451`,
  `${prefix}live segment 3: frames 451..700, mean 12.4, p50 12.0, p95 14.8, skipped 3 draws and 0 dispatches, in force: VICTORY_KEEP_TEXTURES=off VICTORY_TEXTURE_LOW=60 VICTORY_TEXTURE_HIGH=85 VICTORY_LATE_WAITS=off LIVE_SKIP_DISPATCH= LIVE_SKIP_DRAW=1234,5678`,
  `${prefix}not a live line at all`,
].join('\r\n');

test('the segments and changes are read through colour codes and CRLF', () => {
  const segments = parseLiveLog(LOG);
  assert.equal(segments.length, 3);
  // Corrected (the record now says which line shape it read; the older shape has no counted figures).
  assert.deepEqual(segments[0], {
    segment: 1,
    shape: 'old',
    open: false,
    frames: '100..400',
    firstFrame: 100,
    lastFrame: 400,
    frameCount: 301,
    counted: null,
    transitionExcluded: null,
    lightExcluded: null,
    longExcluded: null,
    begun: null,
    ended: null,
    mean: 16.7,
    p50: 16.1,
    p95: 21.3,
    skippedDraws: 0,
    skippedDispatches: 0,
    inForce: {
      VICTORY_KEEP_TEXTURES: 'on',
      VICTORY_TEXTURE_LOW: '70',
      VICTORY_TEXTURE_HIGH: '85',
      VICTORY_LATE_WAITS: 'off',
      LIVE_SKIP_DISPATCH: '',
      LIVE_SKIP_DRAW: '',
    },
    short: false,
    meanChange: null,
    changes: [],
  });
  assert.equal(segments[1].short, true, '50 frames is too few');
  assert.deepEqual(segments[1].changes, [
    { name: 'VICTORY_TEXTURE_LOW', old: '70', new: '60', frame: 401 },
  ]);
  assert.equal(segments[1].meanChange, 3.3);
  assert.deepEqual(
    segments[2].changes.map((change) => [change.name, change.old, change.new]),
    [
      ['LIVE_SKIP_DRAW', '', '1234,5678'],
      ['VICTORY_KEEP_TEXTURES', 'on', 'off'],
    ],
  );
  assert.equal(segments[2].inForce.LIVE_SKIP_DRAW, '1234,5678');
  assert.equal(segments[2].skippedDraws, 3);
  assert.equal(segments[2].meanChange, -7.6);
  assert.equal(SHORT_FRAMES, 120);
});

test('the table names every column, marks a short segment, and shows the change and the switches', () => {
  const text = formatReport(parseLiveLog(LOG));
  // Corrected (a column was added): what the figures stand on, every present for the older line.
  assert.match(text, /\| segment \| frames \| counted \| mean \| p50 \| p95 \| skipped draws \| skipped dispatches \| in force \| mean change \| changed \|/);
  assert.match(text, /\| 1 \| 100\.\.400 \| every present \(older line\) \| 16\.7 \| 16\.1 \| 21\.3 \| 0 \| 0 \|/);
  assert.match(text, /VICTORY_KEEP_TEXTURES=on VICTORY_TEXTURE_LOW=70 .* LIVE_SKIP_DISPATCH= LIVE_SKIP_DRAW= \|/);
  assert.match(text, /\| 2 \| 401\.\.450 \(short\) \| every present \(older line\) \| 20\.0 \| 19\.2 \| 25\.9 \| 0 \| 0 \|/);
  assert.match(text, /\+3\.3 ms/);
  assert.match(text, /VICTORY_TEXTURE_LOW 70 -> 60/);
  assert.match(text, /-7\.6 ms/);
  assert.match(text, /LIVE_SKIP_DRAW \(empty\) -> 1234,5678; VICTORY_KEEP_TEXTURES on -> off/);
  assert.match(text, /short: fewer than 120 frames/);
});

// The runtime's lines as its own tests pin them (fixtures/runtime-lines.mjs), with the timestamp, level and module
// before them: a run's first segment, a change, the next closed by a mark, the plan's first arm begun at the mark,
// and the segment under way.
const LIVE_LOG = [
  `${prefix}${SEGMENT_FIRST_LINE}`,
  `${prefix}${CHANGE_LINE}`,
  `${prefix}${SEGMENT_ENDED_BY_MARK_LINE.replace('live segment 1:', 'live segment 2:')}`,
  `${prefix}${MARK_LINE}`,
  `${prefix}${SEGMENT_BEGUN_AT_MARK_LINE.replace('live segment 2:', 'live segment 3:')}`,
  `${prefix}${SEGMENT_OPEN_LINE.replace('live segment 2 open:', 'live segment 4 open:')}`,
].join('\r\n');

test("the runtime's segment line is read with its three kinds left out, each with its mean, and how it began and ended", () => {
  const closed = parseLiveLine(`${prefix}${SEGMENT_CLOSED_LINE}`).segment;
  assert.deepEqual(
    { ...closed, inForce: undefined },
    {
      segment: 2,
      shape: 'counted',
      open: false,
      frames: '10..19',
      firstFrame: 10,
      lastFrame: 19,
      frameCount: 10,
      counted: 4,
      transitionExcluded: 3,
      transitionFrames: 3,
      transitionMean: 310,
      lightExcluded: 1,
      lightDraws: 1000,
      lightMean: 90,
      longExcluded: 2,
      longMs: 1000,
      longMean: 1500,
      mean: 25,
      p50: 20,
      p95: 40,
      begun: 'change',
      ended: 'change',
      skippedDraws: 0,
      skippedDispatches: 0,
      inForce: undefined,
      short: true,
      meanChange: null,
      changes: [],
    },
  );
  assert.equal(closed.inForce.LIVE_MARK, '0');
  assert.equal(closed.inForce.LIVE_SKIP_DRAW, '', 'an empty list stays empty');
  assert.equal(Object.keys(closed.inForce).length, 14, 'every switch but the panel, off');
  const open = parseLiveLine(SEGMENT_OPEN_LINE).openSegment;
  assert.equal(open.open, true);
  assert.equal(open.ended, null);
  assert.equal(open.mean, 25);
  assert.equal(parseLiveLine(SEGMENT_RUN_END_LINE).segment.ended, 'end');
  assert.equal(parseLiveLine(SEGMENT_TIMER_OPEN_LINE).openSegment.transitionMean, 433.3);
  const first = parseLiveLine(SEGMENT_FIRST_LINE).segment;
  assert.deepEqual([first.begun, first.transitionExcluded, first.transitionMean, first.lightMean, first.longMean], ['start', 0, null, 1500, 1200]);
  const byMark = parseLiveLine(SEGMENT_ENDED_BY_MARK_LINE).segment;
  assert.equal(byMark.ended, 'mark');
  const atMark = parseLiveLine(SEGMENT_BEGUN_AT_MARK_LINE).segment;
  assert.deepEqual([atMark.begun, atMark.transitionExcluded, atMark.mean, atMark.p95], ['mark', 0, 220, 230]);
  assert.deepEqual(parseLiveLine(MARK_LINE).change, { name: 'LIVE_MARK', old: '0', new: '7', frame: 2 });
});

test('a log of the runtime shape: closed segments with the changes that cut them, the mark among them, and the open one', () => {
  const segments = parseLiveLog(LIVE_LOG);
  assert.deepEqual(segments.map((s) => [s.segment, s.begun, s.ended]), [[1, 'start', 'change'], [2, 'start', 'mark'], [3, 'mark', 'change']]);
  assert.deepEqual(segments[1].changes, [{ name: 'VICTORY_TEXTURE_LOW', old: '70', new: '60', frame: 13 }]);
  assert.deepEqual(segments[2].changes, [{ name: 'LIVE_MARK', old: '0', new: '7', frame: 2 }]);
  assert.equal(segments[1].meanChange, 350, '550.0 against 200.0');
  const open = lastOpenSegment(LIVE_LOG);
  assert.equal(open.segment, 4);
  assert.equal(open.counted, 4);
  assert.equal(lastOpenSegment(`${LIVE_LOG}\r\n${prefix}${SEGMENT_CLOSED_LINE.replace('live segment 2:', 'live segment 4:')}`), null, 'closed since');
  const text = formatReport(segments, { open });
  assert.match(text, /\| 3 \| 3\.\.5 \(short\) \| 3 counted; left out 0 transition, 0 light, 0 long; from a mark to a change \| 220\.0 \|/);
  assert.match(text, /\| 4 \(open\) \| 10\.\.19 \(short\) \| 4 counted; left out 3 transition, 1 light, 2 long; from a change to now \(open\) \| 25\.0 \|/);
  assert.match(text, /LIVE_MARK 0 -> 7/);
  assert.match(text, /segment 4 is still open/);
});

test('a segment with nothing counted has no figures, and is short', () => {
  const empty = parseLiveLine(
    SEGMENT_CLOSED_LINE.replace(', 4 counted,', ', 0 counted,').replace('mean 25.0, p50 20.0, p95 40.0', 'mean 0.0, p50 0.0, p95 0.0'),
  ).segment;
  assert.equal(empty.counted, 0);
  assert.equal(empty.mean, null);
  assert.equal(empty.p50, null);
  assert.equal(empty.p95, null);
  assert.equal(empty.short, true);
});

test('a segment of 120 frames is not short; one of 119 is', () => {
  const line = (last) =>
    `${prefix}live segment 1: frames 1..${last}, mean 10.0, p50 10.0, p95 10.0, skipped 0 draws and 0 dispatches, in force: LIVE_SKIP_DRAW=`;
  assert.equal(parseLiveLog(line(120))[0].short, false);
  assert.equal(parseLiveLog(line(119))[0].short, true);
});

test('a log without live lines reports no segments in text and an empty array in JSON', (t) => {
  const dir = tempDir(t);
  const log = path.join(dir, 'run.log');
  fs.writeFileSync(log, `${prefix}a frame line the live report does not use\r\n`);
  assert.equal(formatReport(parseLiveLog(fs.readFileSync(log, 'utf8'))), 'no live segments');
  const result = spawnSync(process.execPath, [script, log, '--json'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), []);
});

test('the command prints the same rows as objects with --json', (t) => {
  const dir = tempDir(t);
  const log = path.join(dir, 'run.log');
  fs.writeFileSync(log, LOG);
  const text = spawnSync(process.execPath, [script, log], { encoding: 'utf8' });
  assert.equal(text.status, 0, text.stderr);
  assert.match(text.stdout, /\| 2 \| 401\.\.450 \(short\) \|/);
  const json = spawnSync(process.execPath, [script, '--json', log], { encoding: 'utf8' });
  assert.equal(json.status, 0, json.stderr);
  const segments = JSON.parse(json.stdout);
  assert.equal(segments.length, 3);
  assert.equal(segments[1].short, true);
  assert.equal(segments[1].meanChange, 3.3);
  assert.deepEqual(segments[2].changes.map((change) => change.name), ['LIVE_SKIP_DRAW', 'VICTORY_KEEP_TEXTURES']);
});

test('the command refuses with usage when the log is missing', () => {
  const result = spawnSync(process.execPath, [script], { encoding: 'utf8' });
  assert.equal(result.status, 2);
  assert.match(result.stderr, /usage: node live-report\.mjs <log> \[--json\]/);
});
