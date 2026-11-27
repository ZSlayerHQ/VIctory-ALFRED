import assert from 'node:assert/strict';
import test from 'node:test';

import { sampleObject } from './fixtures/old-shape.mjs';
import { RUNTIME_PHASES, STATS_ALL_OFF, STATS_LIVE, STATS_OPTIONAL_PARTS, statsObject, statsWith } from './fixtures/runtime-stats.mjs';
import { objectAgeMs, objectsFigures, parseStatsText, readStatsObject, sequenceOf } from './stats-object.mjs';

test("the runtime's object with every switch off is read by its names, none of them null", () => {
  const record = parseStatsText(STATS_ALL_OFF);
  assert.equal(record.shape, 'interval');
  assert.equal(record.seq, 7);
  assert.equal(record.tMs, 7000);
  assert.equal(record.unixMs, 1800000007000);
  assert.deepEqual(record.interval, { firstFrame: 1230, lastFrame: 1234, frames: 5, wallMs: 1300 });
  assert.equal(record.frame, 1234);
  assert.equal(record.fps, 3.85);
  assert.equal(record.frameMs, 260);
  assert.equal(record.frameMsP50, 220);
  assert.equal(record.frameMsWorst, 420, 'frame_ms_worst, the longest of the object');
  assert.deepEqual(record.frameMsList, [220, 220, 420, 220, 220]);
  assert.equal(record.frameMsListMissing, 0);
  assert.equal(record.budget.frames, 2);
  assert.equal(record.budget.frameMs, 200);
  assert.equal(record.budget.calls, 0);
  assert.equal(record.budget.largestTitleGapMs, 0);
  assert.equal(record.budget.titleAndCallsCpuMs, 0);
  assert.deepEqual(record.budget.phases.title, { ms: 100, cpuMs: null }, 'title has no processor time of its own');
  assert.deepEqual(record.budget.phases.relay_wait, { ms: 80, cpuMs: 40 });
  assert.deepEqual(record.otherThreads.phases.relay_land, { ms: 60, cpuMs: 15 });
  assert.equal(record.summed.host_wait_ms, 78);
  assert.equal(record.gpuStamps, null);
  assert.equal(record.draws, 6500);
  assert.equal(record.dispatched, 12);
  assert.equal(record.dropped, 2);
  assert.equal(record.storesFailed, 0);
  assert.equal(record.storesFailedBytes, 0);
  assert.equal(record.pipelinesCreated, 0);
  assert.equal(record.pipelinesPerMinute, 0);
  assert.equal(record.vramMb, 18600);
  assert.equal(record.live.VICTORY_EXACT_BACK, 'on');
  assert.equal(record.segmentsClosed, 3, 'segments_closed_total');
  assert.equal(record.segmentOpen, null);
  assert.equal(record.kernelCalls.perFrame.reads, 0);
  assert.equal(record.kernelCalls.raw.reads_total, 0);
  assert.deepEqual(record.optionalOn, []);
  // Every field the record names is read from the real object: none is null for want of a name.
  const nullable = new Set(['segmentOpen', 'gpuStamps', 'dispatchesSincePrevious', 'hashCheck', 'textureReceipts', 'landedClasses', 'textureRecheck', 'texturesMade']);
  for (const [field, value] of Object.entries(record)) {
    if (!nullable.has(field)) assert.notEqual(value, null, `${field} is null on the runtime's own object`);
  }
});

test("the live run's object: the open segment's three kinds left out, the mark among the begun, the budget's calls", () => {
  const record = parseStatsText(STATS_LIVE);
  assert.equal(record.segmentsClosed, 1);
  assert.deepEqual(record.segmentOpen, {
    segment: 2,
    firstFrame: 10,
    lastFrame: 19,
    counted: 4,
    transitionExcluded: 3,
    transitionMeanMs: 310,
    lightExcluded: 1,
    lightMeanMs: 90,
    longExcluded: 2,
    longMeanMs: 1500,
    meanMs: 25,
    p50Ms: 20,
    p95Ms: 40,
    begun: 'change',
  });
  assert.equal(record.budget.calls, 2000);
  assert.equal(record.budget.largestTitleGapMs, 3.5);
  assert.equal(record.budget.titleAndCallsCpuMs, 90);
  for (const name of ['list_calls', 'object_calls', 'hle_calls']) assert.equal(record.budget.phases[name].cpuMs, null, name);
  assert.equal(record.live.LIVE_MARK, '0', 'LIVE_MARK is in the live map');
  assert.equal(record.gpuStamps.copy_queue.fetch_lists, 2);
  assert.match(record.gpuStamps.idle, /^not derivable/);
  const mark = readStatsObject(statsObject({ segment_open: { ...JSON.parse(STATS_LIVE).segment_open, begun: 'mark' } }));
  assert.equal(mark.segmentOpen.begun, 'mark');
  // Every phase the object writes is one of the runtime's.
  for (const name of Object.keys(record.budget.phases)) assert.ok(RUNTIME_PHASES.includes(name), name);
});

test('the pre-rebuild sample is read by its old names, and what it lacks is null, never zero', () => {
  const record = readStatsObject(sampleObject());
  assert.equal(record.shape, 'sample');
  assert.match(record.shapeName, /pre-rebuild/);
  assert.equal(record.seq, null);
  assert.equal(record.unixMs, null);
  assert.equal(record.interval, null);
  assert.equal(record.frameMsP50, null);
  assert.equal(record.frameMsWorst, null);
  assert.equal(record.frameMsList, null);
  assert.equal(record.budget, null);
  assert.equal(record.segmentOpen, null);
  assert.equal(record.segmentsClosed, 1, 'the sample called the closed segments `segment`');
  assert.equal(record.summed.host_wait_ms, 6, '`parts` is read as the summed figures');
  assert.equal(record.dispatched, null, "the old count since the previous object is not a per-frame figure");
  assert.equal(record.dispatchesSincePrevious, 1860);
  assert.equal(record.pipelinesCreated, 900);
  assert.equal(objectAgeMs(record), null, 'no clock of its own: the caller uses the file time');
});

test('an optional part absent means its switch is off, never zeros; each one on is read and named', () => {
  const off = parseStatsText(STATS_ALL_OFF);
  for (const field of ['hashCheck', 'textureReceipts', 'landedClasses', 'textureRecheck', 'texturesMade']) assert.equal(off[field], null, field);
  for (const name of Object.keys(STATS_OPTIONAL_PARTS)) assert.deepEqual(parseStatsText(statsWith(name)).optionalOn, [name]);
  assert.deepEqual(parseStatsText(statsWith('hash_check')).hashCheck, { missedPiecesTotal: 0, checkedPiecesTotal: 0, skippedTotal: 0 });
  assert.equal(parseStatsText(statsWith('texture_receipts')).textureReceipts.installs_total, 0);
  assert.equal(parseStatsText(statsWith('landed_classes')).landedClasses, null, 'on, but no 600-frame window has ended');
  assert.equal(parseStatsText(statsWith('texture_recheck')).textureRecheck.stale_total, 0);
  assert.equal(parseStatsText(statsWith('textures_made')).texturesMade.last_minute, null);
});

test('a torn or foreign text is not an object; a segment with nothing counted has no figures', () => {
  assert.equal(parseStatsText('{"frame": 12, "fps": '), null);
  assert.equal(parseStatsText(STATS_ALL_OFF.slice(0, 2000)), null, 'half an object');
  assert.equal(parseStatsText('[1,2]'), null);
  assert.equal(parseStatsText('{"fps": 3}'), null);
  const open = JSON.parse(STATS_LIVE).segment_open;
  const empty = readStatsObject(statsObject({ segment_open: { ...open, counted: 0, mean_ms: 0, p50_ms: 0, p95_ms: 0, light_excluded: 0 } }));
  assert.equal(empty.segmentOpen.meanMs, null);
  assert.equal(empty.segmentOpen.p95Ms, null);
  assert.equal(empty.segmentOpen.lightMeanMs, null, 'no light frame: its mean 0.00 is no figure');
  assert.equal(readStatsObject(statsObject({ segment_open: null })).segmentOpen, null);
});

test("a fixture copy refuses a name the runtime does not write, so a test cannot bring back the first shape's", () => {
  assert.throws(() => statsObject({ segment: 3 }), /not a field/);
  assert.throws(() => statsObject({ frame_ms_p95: 3 }), /not a field/);
  assert.throws(() => statsObject({ segment_open: { excluded: 3 } }), /segment_open.excluded/);
});

test('seq tells a repeated read, a missed object and a new run; unix_ms tells a stale one', () => {
  const at = (seq) => readStatsObject(statsObject({ seq }));
  assert.deepEqual(sequenceOf(null, at(5)), { kind: 'first', missed: 0 });
  assert.deepEqual(sequenceOf(at(5), at(5)), { kind: 'repeat', missed: 0 });
  assert.deepEqual(sequenceOf(at(5), at(6)), { kind: 'next', missed: 0 });
  assert.deepEqual(sequenceOf(at(5), at(9)), { kind: 'missed', missed: 3 });
  assert.deepEqual(sequenceOf(at(9), at(2)), { kind: 'older', missed: 0 });
  assert.equal(sequenceOf(at(5), readStatsObject(sampleObject())), null);
  assert.equal(objectAgeMs(parseStatsText(STATS_ALL_OFF), 1800000011000), 4000);
});

test('an arm over objects weights each mean by its frames and takes percentiles from the joined lists', () => {
  const a = readStatsObject(statsObject({ interval: { first_frame: 1, last_frame: 2, frames: 2, wall_ms: 20 }, frame_ms: 10, frame_ms_list: [10, 10] }));
  const b = readStatsObject(
    statsObject({ interval: { first_frame: 3, last_frame: 10, frames: 8, wall_ms: 320 }, frame_ms: 40, frame_ms_list: [40, 40, 40, 40, 40, 40, 40, 100] }),
  );
  const figures = objectsFigures([a, b]);
  assert.equal(figures.shape, 'interval');
  assert.equal(figures.frames, 10);
  assert.equal(figures.meanMs, 34, '(2 x 10 + 8 x 40) / 10, not the plain mean of 25');
  assert.equal(figures.p50Ms, 40);
  assert.equal(figures.p95Ms, 100);
  assert.equal(figures.listFrames, 10);
  const old = objectsFigures([readStatsObject(sampleObject({ frame_ms: 10 })), readStatsObject(sampleObject({ frame_ms: 30 }))]);
  assert.equal(old.shape, 'sample');
  assert.equal(old.meanMs, 20);
  assert.equal(old.p95Ms, null, 'no frame list in the older shape: n/a');
  assert.equal(old.frames, null);
});
