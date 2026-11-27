import test from 'node:test';
import assert from 'node:assert/strict';
import * as overlay from './live-overlay.mjs';
import { readStatsObject } from './stats-object.mjs';
import { sampleObject } from './fixtures/old-shape.mjs';
import { statsObject } from './fixtures/runtime-stats.mjs';

const { panel, step, switchesOf } = overlay;
// Reached through the module so a missing export fails only its own test.
const isActive = (...args) => overlay.isActive(...args);

const sample = (frame, frameMs, live) => ({ frame, fps: 1000 / frameMs, frame_ms: frameMs, live });

test('switches are read from an object, from pairs and from named entries', () => {
  assert.deepEqual(switchesOf({ live: { A: 'on', B: 3 } }), { A: 'on', B: '3' });
  assert.deepEqual(switchesOf({ live: [['A', 'on'], ['B', 'off']] }), { A: 'on', B: 'off' });
  assert.deepEqual(switchesOf({ live: [{ name: 'A', value: 'on' }] }), { A: 'on' });
  assert.deepEqual(switchesOf({}), {});
});

test('the first sample sets the switches without reporting a change', () => {
  const state = step({}, sample(1, 200, { A: 'off' }), 1000);
  assert.deepEqual(state.switches, { A: 'off' });
  assert.equal(state.changes.length, 0);
});

test('a change is reported with the frame time before it and, as samples arrive, after it', () => {
  let state = {};
  state = step(state, sample(1, 200, { A: 'off' }), 1000);
  state = step(state, sample(2, 220, { A: 'off' }), 2000);
  state = step(state, sample(3, 150, { A: 'on' }), 3000);
  state = step(state, sample(4, 170, { A: 'on' }), 4000);
  const shown = panel(state, sample(4, 170, { A: 'on' }), 4000, true);
  assert.equal(shown.changes.length, 1);
  assert.deepEqual(
    { name: shown.changes[0].name, from: shown.changes[0].from, to: shown.changes[0].to, frame: shown.changes[0].frame },
    { name: 'A', from: 'off', to: 'on', frame: 3 },
  );
  assert.equal(shown.changes[0].before_ms, 210);
  assert.equal(shown.changes[0].after_ms, 160);
  assert.equal(shown.changes[0].after_samples, 2);
  assert.deepEqual(shown.switches, [{ name: 'A', value: 'on' }]);
});

test('the same sample read twice is counted once, and a switch that disappears reads as unset', () => {
  let state = step({}, sample(1, 200, { A: 'on' }), 1000);
  state = step(state, sample(1, 200, { A: 'on' }), 2000);
  assert.equal(state.recent.length, 1);
  state = step(state, sample(2, 200, {}), 3000);
  assert.equal(state.changes[0].to, '(unset)');
});

test('an idle panel says so and keeps no figures', () => {
  const shown = panel({}, null, 5000, false);
  assert.equal(shown.active, false);
  assert.equal(shown.frame_ms, null);
  assert.equal(shown.stats_shape, null);
  assert.deepEqual(shown.switches, []);
});

// The runtime's object (fixtures/runtime-stats.mjs): one covers every frame since the one before it.
const intervalObject = (overrides = {}) => statsObject(overrides);
const interval = (seq, frames, frameMs, live, extra = {}) =>
  intervalObject({
    seq,
    frame: 1000 + seq * 10,
    interval: { first_frame: 1000 + seq * 10 - frames + 1, last_frame: 1000 + seq * 10, frames, wall_ms: frames * frameMs },
    frame_ms: frameMs,
    frame_ms_p50: frameMs,
    frame_ms_worst: frameMs,
    frame_ms_list: Array.from({ length: frames }, () => frameMs),
    live,
    ...extra,
  });

test("the panel reads frame_ms, p50 and the worst frame from either shape and says which it read; the older shape has neither", () => {
  const now = intervalObject().unix_ms;
  const fresh = panel(step({}, intervalObject(), now), intervalObject(), now, true);
  assert.equal(fresh.stats_shape, 'interval');
  assert.equal(fresh.frame_ms, 260);
  assert.equal(fresh.frame_ms_p50, 220);
  assert.equal(fresh.frame_ms_worst, 420, "the runtime's frame_ms_worst");
  assert.equal(fresh.frame, 1234);
  assert.equal('frame_ms_p95' in fresh, false, 'no 95th percentile of an object: the runtime writes none');
  const older = panel(step({}, sampleObject(), now), sampleObject(), now, true);
  assert.equal(older.stats_shape, 'sample');
  assert.equal(older.frame_ms, 250);
  assert.equal(older.frame_ms_p50, null, 'n/a, never zero');
  assert.equal(older.frame_ms_worst, null, 'n/a, never zero');
});

test('a repeated seq is not pushed twice, a seq jump counts the objects missed, and a lower seq starts afresh', () => {
  let state = step({}, interval(5, 5, 200, { A: 'off' }), 1000);
  state = step(state, interval(5, 5, 200, { A: 'off' }), 2000);
  assert.equal(state.recent.length, 1, 'the same object read twice');
  state = step(state, interval(8, 5, 200, { A: 'on' }), 3000);
  assert.equal(state.recent.length, 2);
  assert.equal(state.missed, 2, 'seq 6 and 7 were written and never read');
  assert.equal(state.changes.length, 1);
  assert.equal(panel(state, interval(8, 5, 200, { A: 'on' }), 3000, true).objects_missed, 2);
  // A lower seq in the same file: the runtime started again, so the last run's changes are not this one's.
  state = step(state, interval(1, 5, 300, { A: 'off' }), 4000);
  assert.equal(state.changes.length, 0);
  assert.equal(state.recent.length, 1);
  assert.equal(state.missed, 0);
});

test('before and after a change, interval objects are weighted by the frames each covers', () => {
  let state = {};
  // A pause is one object holding the whole gap: one 1000 ms frame, then nine of 100 ms.
  state = step(state, interval(1, 1, 1000, { A: 'off' }), 1000);
  state = step(state, interval(2, 9, 100, { A: 'off' }), 2000);
  state = step(state, interval(3, 2, 50, { A: 'on' }), 3000);
  state = step(state, interval(4, 8, 75, { A: 'on' }), 4000);
  const shown = panel(state, interval(4, 8, 75, { A: 'on' }), 4000, true);
  assert.equal(shown.changes[0].before_ms, 190, '(1000 + 9 x 100) / 10 frames, not the mean of two objects (550)');
  assert.equal(shown.changes[0].after_ms, 70, '(2 x 50 + 8 x 75) / 10 frames');
  assert.equal(shown.changes[0].after_samples, 2);
  assert.equal(shown.changes[0].after_frames, 10);
});

test('a stats object is live by its own clock (unix_ms) when it has one, else by the file time', () => {
  const now = 1_795_000_100_000;
  // Written 30 s ago by its own clock, though the file was touched just now: stale.
  assert.equal(isActive(readStatsObject(intervalObject({ unix_ms: now - 30_000 })), now, now), false);
  assert.equal(isActive(readStatsObject(intervalObject({ unix_ms: now - 2_000 })), now - 60_000, now), true);
  // The older shape has no clock: the file time decides.
  assert.equal(isActive(readStatsObject(sampleObject()), now - 2_000, now), true);
  assert.equal(isActive(readStatsObject(sampleObject()), now - 30_000, now), false);
});
