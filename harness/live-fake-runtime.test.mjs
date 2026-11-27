// The fake runtime writes what the runtime writes: given the frames of the runtime's own tests, its writers give the
// fixtures (copied from those tests) character for character, and a running fake writes only those shapes.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  FRAME_LINE,
  SEGMENT_BEGUN_AT_MARK_LINE,
  SEGMENT_CLOSED_LINE,
  SEGMENT_ENDED_BY_MARK_LINE,
  SEGMENT_FIRST_LINE,
  SEGMENT_OPEN_LINE,
  SEGMENT_RUN_END_LINE,
  SEGMENT_TIMER_OPEN_LINE,
} from './fixtures/runtime-lines.mjs';
import { IN_FORCE_DEFAULTS, STATS_LIVE } from './fixtures/runtime-stats.mjs';
import { DEFAULT_START, fakeRuntime, frameLineOver, frameModel, listedLive, makeSim, segmentFigures, segmentLine, segmentOpenObject, statsObjectOver } from './live-fake-runtime.mjs';
import { parseLiveLine } from './live-report.mjs';
import { parseStatsText } from './stats-object.mjs';

const frames = (list) => list.map(([ms, draws = 7000]) => ({ ms, draws }));
const line = (k, first, last, list, begun, ended) =>
  segmentLine({ k, first, last, figures: segmentFigures(frames(list), begun), begun, ended, inForce: IN_FORCE_DEFAULTS });

test("the segment writer gives the runtime's lines for the frames of the runtime's tests", () => {
  // live.rs the_marker_and_segment_lines_have_their_fixed_shape: frames 10..19 after a change.
  const fixed = [[300], [320], [310], [90, 20], [1500], [1500], [10], [20], [30], [40]];
  assert.equal(line(2, 10, 19, fixed, 'change', null), SEGMENT_OPEN_LINE);
  assert.equal(line(2, 10, 19, fixed, 'change', 'change'), SEGMENT_CLOSED_LINE);
  // a_segment_leaves_its_transition_light_and_long_frames_out_and_says_so: the run's first segment.
  assert.equal(line(1, 1, 4, [[1500, 20], [200], [200], [1200]], 'start', 'change'), SEGMENT_FIRST_LINE);
  // a_mark_cuts_a_segment_and_leaves_no_frame_out.
  assert.equal(line(1, 1, 2, [[900], [200]], 'start', 'mark'), SEGMENT_ENDED_BY_MARK_LINE);
  assert.equal(line(2, 3, 5, [[210], [220], [230]], 'mark', 'change'), SEGMENT_BEGUN_AT_MARK_LINE);
  // a_run_ended_by_the_timer_has_its_last_segment_written.
  const timer = [[700], [300], [300], [240], [236], [244]];
  assert.equal(line(2, 2, 7, timer, 'change', null), SEGMENT_TIMER_OPEN_LINE);
  assert.equal(line(2, 2, 7, timer, 'change', 'end'), SEGMENT_RUN_END_LINE);
});

test("the object writer keeps the runtime's keys in their order, and its segment_open is the runtime's", () => {
  const live = JSON.parse(STATS_LIVE);
  const figures = segmentFigures(frames([[300], [320], [310], [90, 20], [1500], [1500], [10], [20], [30], [40]]), 'change');
  const open = segmentOpenObject({ k: 2, first: 10, last: 19, figures, begun: 'change' });
  assert.equal(JSON.stringify(open), JSON.stringify(live.segment_open));
  const object = statsObjectOver({ seq: 3, tMs: 3000, unixMs: 1, firstFrame: 1230, times: [220, 220, 420, 220, 220], live: listedLive(DEFAULT_START), closed: 1, open });
  assert.deepEqual(Object.keys(object), Object.keys(live));
  assert.deepEqual(Object.keys(object.live), Object.keys(live.live), "the registry's order, LIVE_MARK in it");
  const record = parseStatsText(JSON.stringify(object));
  assert.deepEqual([record.frameMsP50, record.frameMsWorst, record.segmentsClosed, record.interval.frames], [220, 420, 1, 5]);
  assert.equal(
    frameLineOver({ first: 1, last: 6, wallMs: 2000, p50: 250, p95: 500 }),
    FRAME_LINE,
    "the frame line for the frames of a_frame_line_covers_every_second_since_the_last is the runtime's",
  );
});

function tempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'live-fake-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

test('a running fake writes only the runtime shapes: every live line parses, every object has the fixture keys', async (t) => {
  const dir = tempDir(t);
  const log = path.join(dir, 'run.log');
  const control = path.join(dir, 'live.txt');
  const stats = path.join(dir, 'stats.json');
  const fake = fakeRuntime(t, { log, control, statsAt: stats, tickMs: 5 });
  const keys = Object.keys(JSON.parse(STATS_LIVE));
  await fake.clock.sleep(60);
  assert.deepEqual(Object.keys(JSON.parse(fs.readFileSync(stats, 'utf8'))), keys);
  fake.handSet('LIVE_MARK', '5');
  await fake.clock.sleep(60);
  fake.handSet('VICTORY_TEXTURE_LOW', '60');
  await fake.clock.sleep(60);
  fake.shutdown();
  const lines = fs.readFileSync(log, 'utf8').split('\n').filter((text) => text.includes('live segment') || text.startsWith('live:'));
  for (const text of lines) assert.ok(parseLiveLine(text), `the fake wrote a line no reader parses: ${text}`);
  const closed = lines.map(parseLiveLine).filter((parsed) => parsed.segment).map((parsed) => parsed.segment);
  assert.deepEqual(closed.map((s) => [s.begun, s.ended]), [['start', 'mark'], ['mark', 'change'], ['change', 'end']]);
  assert.equal(closed[1].transitionExcluded, 0, 'a mark leaves no frame out');
  assert.ok(closed[2].transitionExcluded > 0, 'a change leaves its first frames out');
  const object = parseStatsText(fs.readFileSync(stats, 'utf8'));
  assert.equal(object.segmentsClosed, 2);
  assert.equal(object.live.LIVE_MARK, '5');

  const sim = makeSim({ dir, name: 'sim', frameMs: () => 230 });
  sim.advanceTo(8000);
  sim.handSet({ LIVE_MARK: '1' });
  sim.advanceTo(16000);
  sim.shutdown();
  const simLines = fs.readFileSync(sim.log, 'utf8').split('\n').filter((text) => text.includes('live segment'));
  for (const text of simLines) assert.ok(parseLiveLine(text), text);
  assert.deepEqual(Object.keys(JSON.parse(fs.readFileSync(sim.stats, 'utf8'))), keys);
});

test('the simulated title reads a control-file write that kept the file time of the one before (a coarse clock)', (t) => {
  // The hosted runner's failure: on the virtual clock two writes fall inside one step of a coarse file-system
  // timestamp, and a sim that read the file by its modified time never saw the second ('timed out after 15 s
  // waiting for the game to log ...').
  const dir = tempDir(t);
  const sim = makeSim({ dir, seed: 1, frameMs: frameModel({ noise: 'low' }) });
  sim.advanceTo(2000);
  // Both writes carry one whole-second time, as a coarse clock gives two writes a few real milliseconds apart.
  const step = new Date(Math.floor(Date.now() / 1000) * 1000);
  fs.writeFileSync(sim.control, 'VICTORY_FETCH_QUEUE=on\n');
  fs.utimesSync(sim.control, step, step);
  sim.advanceTo(4000);
  assert.equal(sim.values.VICTORY_FETCH_QUEUE, 'on');
  fs.writeFileSync(sim.control, 'VICTORY_FETCH_QUEUE=off\n');
  fs.utimesSync(sim.control, step, step);
  sim.advanceTo(6000);
  assert.equal(sim.values.VICTORY_FETCH_QUEUE, 'off', 'the second write is read though its file time did not move');
  assert.deepEqual(
    sim.markers.map((marker) => `${marker.name} ${marker.old} -> ${marker.new}`),
    ['VICTORY_FETCH_QUEUE off -> on', 'VICTORY_FETCH_QUEUE on -> off'],
  );
  sim.shutdown();
});
