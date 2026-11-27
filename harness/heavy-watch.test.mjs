import assert from 'node:assert/strict';
import test from 'node:test';

import { decide, endedMarker } from './heavy-watch.mjs';

// A VIctory 5-second frame line, the shape FRAME_RE in parse-run.mjs matches.
function frameLine(ts, draws, dropped = 0) {
  return `${ts}  INFO victory_gpu::host: 30.0 fps, 33.3 ms/frame (0.3 preparing: copies 0.2 ms 0.4 MB, textures 0.0 ms 0.0 MB; 0.7 on the GPU; 0.0 copying back 0.0 MB), ${draws} draws/frame, dropped ${dropped} without a host pipeline (0 being made) and 1 indirect`;
}

const at = (seconds) => Date.parse(`2026-11-24T12:00:${String(seconds).padStart(2, '0')}.000000Z`);

test('heavy watcher: not yet heavy does nothing, however long the run has gone', () => {
  const lines = [frameLine('2026-11-24T12:00:00.000000Z', 900, 99)];
  assert.deepEqual(decide(lines, { seconds: 10, now: at(59), gameAlive: true }), {
    action: 'wait',
    trigger: null,
    elapsedSeconds: null,
  });
});

test('heavy watcher: heavy for less than N waits, counting from the trigger line timestamp', () => {
  const lines = [
    frameLine('2026-11-24T12:00:00.000000Z', 13),
    frameLine('2026-11-24T12:00:10.000000Z', 145, 2744), // 2889 draws+dropped: heavy
  ];
  const decision = decide(lines, { seconds: 10, now: at(19), gameAlive: true });
  assert.equal(decision.action, 'wait');
  assert.equal(decision.elapsedSeconds, 9);
  assert.deepEqual(decision.trigger, { index: 1, draws: 145, dropped: 2744 });
});

test('heavy watcher: heavy for N ends, including exactly N', () => {
  const lines = [frameLine('2026-11-24T12:00:10.000000Z', 1000, 50)];
  const decision = decide(lines, { seconds: 10, now: at(20), gameAlive: true });
  assert.equal(decision.action, 'end');
  assert.equal(decision.elapsedSeconds, 10);

  // A later poll need not carry the trigger line again: the watcher keeps its timestamp.
  const later = decide([], { seconds: 10, now: at(21), gameAlive: true, triggerAt: at(10) });
  assert.equal(later.action, 'end');
  assert.equal(later.elapsedSeconds, 11);
});

test('heavy watcher: a game that is already gone is never ended, even when N has passed', () => {
  const heavy = [frameLine('2026-11-24T12:00:10.000000Z', 1000, 50)];
  assert.equal(decide(heavy, { seconds: 10, now: at(30), gameAlive: false }).action, 'gone');
  assert.equal(decide([], { seconds: 10, now: at(30), gameAlive: false }).action, 'gone');
  // Exactly one heavier line before the game goes is enough to arm the timer, not to end it.
  const light = [frameLine('2026-11-24T12:00:00.000000Z', 900, 99)];
  assert.equal(decide(light, { seconds: 10, now: at(30), gameAlive: false }).action, 'gone');
});

test('heavy watcher: the log line the parser reads says N exactly', () => {
  assert.equal(
    endedMarker(30),
    'harness: ended by --heavy-seconds after 30 s of heavy scene',
  );
});
