import assert from 'node:assert/strict';
import test from 'node:test';
import {
  DEFAULT_FLOOR_FPS,
  DEFAULT_TIMEOUT_S,
  DEFAULT_WAITING,
  SETTLED_LINES,
  due,
  PIPELINE_QUIET_MS,
  frameLine,
  initialState,
  observe,
  settleEndMarker,
  settledMarker,
  timeoutMarker,
  worldTimeoutMarker,
  UNSETTLED_NOTE_MS,
  unmetText,
  unsettledMarker,
} from './settled-watch.mjs';

// Frame lines in the runtime's format, cut after the texture queue's count, with made-up values.
const EARLY =
  '2026-11-20T10:17:12.000000Z  INFO victory_gpu::host: 30.0 fps, 33.3 ms/frame (1.6 preparing: copies 1.4 ms 0.3 MB, textures 0.0 ms 0.0 MB; 0.0 on the GPU; 0.0 copying back 0.0 MB), 12 draws/frame, dropped 0 without a host pipeline (900 being made) and 0 indirect; heap buffers 0.0 passes 0.0 ms;                  texture slots written so far: 1000 placed, 5000 of objects; textures picked by index: 500 uploaded so far (200 MB), 0 waiting, 0 passed over unmapped;';
const HEAVY_LOADING =
  '2026-11-20T10:18:50.000000Z  INFO victory_gpu::host: 0.8 fps, 1250.0 ms/frame (600.0 preparing: copies 150.0 ms 240.0 MB, textures 420.0 ms 120.0 MB; 30.0 on the GPU; 100.0 copying back 70.0 MB), 1600 draws/frame, dropped 900 without a host pipeline (0 being made) and 10 indirect; heap buffers 0.4 passes 350.0 ms;                  texture slots written so far: 150000 placed, 30000 of objects; textures picked by index: 35000 uploaded so far (2900 MB), 1200 waiting, 1300 passed over unmapped;';
const PLAYING =
  '2026-11-20T10:19:47.000000Z  INFO victory_gpu::host: 2.6 fps, 384.6 ms/frame (180.0 preparing: copies 140.0 ms 190.0 MB, textures 15.0 ms 40.0 MB; 45.0 on the GPU; 85.0 copying back 75.0 MB), 1400 draws/frame, dropped 600 without a host pipeline (0 being made) and 0 indirect; heap buffers 0.6 passes 210.0 ms;                  texture slots written so far: 260000 placed, 70000 of objects; textures picked by index: 77000 uploaded so far (6400 MB), 1500 waiting, 6700 passed over unmapped;';
const later = (line, mmss) => line.replace(/T\d\d:\d\d:\d\d\.\d+Z/, `T10:${mmss}.000000Z`);

test('a frame line with the runtime frame-number prefix parses the same figures', () => {
  const line =
    '2026-11-26T05:02:11.250000Z  INFO victory_gpu::host: frame 18432, 27.4 fps, 36.5 ms/frame ' +
    '(0.4 preparing: copies 0.3 ms 1.2 MB, textures hashed 0.2 ms 0.1 MB; 1.7 on the GPU; ' +
    '0.1 copying back 0.9 MB), 5000 draws/frame, dropped 3 without a host pipeline (1 being made) and 7 indirect';
  assert.deepEqual(frameLine(line), {
    at: Date.parse('2026-11-26T05:02:11.250000Z'),
    fps: 27.4,
    draws: 5000,
    dropped: 3,
    making: 1,
    waiting: null,
  });
});

test('a frame line gives its figures, or nothing for another line', () => {
  assert.deepEqual(frameLine(HEAVY_LOADING), {
    at: Date.parse('2026-11-20T10:18:50.000000Z'),
    fps: 0.8,
    draws: 1600,
    dropped: 900,
    making: 0,
    waiting: 1200,
  });
  assert.equal(frameLine('2026-11-20T10:18:50Z  INFO victory_gpu::pipeline: shader translated stage="pixel"'), null);
  assert.equal(frameLine('not a log line'), null);
});

test('the rate is the test: three consecutive heavy lines at or above the floor, whatever the queue holds', () => {
  assert.equal(DEFAULT_FLOOR_FPS, 1.5);
  assert.equal(DEFAULT_WAITING, null, 'no waiting ceiling unless one is asked for');
  assert.equal(SETTLED_LINES, 3);
  let state = initialState();
  for (const line of [EARLY, HEAVY_LOADING, PLAYING, later(PLAYING, '20:57')]) state = observe(state, line);
  assert.equal(state.heavyAt, Date.parse('2026-11-20T10:18:50.000000Z'));
  assert.equal(state.consecutive, 2);
  assert.equal(state.settledAt, null, 'two lines are not three');
  state = observe(state, later(PLAYING, '21:07'));
  assert.equal(state.settledAt, Date.parse('2026-11-20T10:21:07.000000Z'));
  assert.equal(state.settledWaiting, 1500, 'the queue count is reported, not required');
  assert.equal(state.settledFps, 2.6);
});

test('a waiting ceiling is optional and only applies when asked for', () => {
  let state = initialState();
  for (const line of [PLAYING, later(PLAYING, '20:57'), later(PLAYING, '21:07')]) state = observe(state, line, { waiting: 100 });
  assert.equal(state.settledAt, null, '1500 waiting is above a ceiling of 100');
  state = initialState();
  for (const line of [PLAYING, later(PLAYING, '20:57'), later(PLAYING, '21:07')]) state = observe(state, line, { waiting: 1600 });
  assert.notEqual(state.settledAt, null);
});

test('the marker says what the settle saw, and the clock ends or times out as asked', () => {
  let state = initialState();
  state = observe(state, HEAVY_LOADING);
  const heavy = state.heavyAt;
  assert.equal(due(state, { now: heavy + 1000, seconds: 75, timeout: DEFAULT_TIMEOUT_S }), 'wait');
  assert.equal(due(state, { now: heavy + 601_000, seconds: 75, timeout: 600 }), 'timeout');
  assert.equal(timeoutMarker(600), 'harness: ended by --settled-timeout after 600 s: the scene never settled');

  for (const line of [PLAYING, later(PLAYING, '20:57'), later(PLAYING, '21:07')]) state = observe(state, line);
  const since = Math.round((state.settledAt - state.heavyAt) / 1000);
  assert.equal(since, 137);
  assert.equal(settledMarker(since, state.settledWaiting, state.settledFps), 'harness: settled at 137 s (waiting 1500, 2.6 fps)');
  const settled = state.settledAt;
  assert.equal(due(state, { now: settled + 74_000, seconds: 75, timeout: 600 }), 'wait');
  assert.equal(due(state, { now: settled + 75_000, seconds: 75, timeout: 600 }), 'end');
  assert.equal(settleEndMarker(75), 'harness: ended by --settled-seconds after 75 s of settled scene');
  assert.equal(settledMarker(5, null, 3), 'harness: settled at 5 s (waiting ?, 3 fps)');
});

test('a run that never reaches the world ends at --world-timeout, counted from the watcher start', () => {
  const start = Date.parse('2026-11-20T10:00:00.000000Z');
  let state = initialState();
  assert.equal(
    due(state, { now: start + 299_000, seconds: 75, timeout: 600, worldTimeout: 300, startedAt: start }),
    'wait',
  );
  assert.equal(
    due(state, { now: start + 300_000, seconds: 75, timeout: 600, worldTimeout: 300, startedAt: start }),
    'world-timeout',
  );
  assert.equal(due(state, { now: start + 3_600_000, seconds: 75, timeout: 600 }), 'wait', 'without the flag nothing changes');

  state = observe(state, later(EARLY, '00:05'));
  assert.equal(state.heavyAt, null, 'menu-level draws are not a heavy line');
  assert.match(
    unmetText(state, { worldTimeout: 300, startedAt: start, now: start + 60_000 }),
    /first heavy frame line; 240 s left/,
  );
  assert.equal(
    due(state, { now: start + 300_000, seconds: 75, timeout: 600, worldTimeout: 300, startedAt: start }),
    'world-timeout',
  );

  state = observe(state, later(PLAYING, '00:10'));
  assert.notEqual(state.heavyAt, null);
  assert.equal(
    due(state, { now: start + 3_600_000, seconds: 75, timeout: 600, worldTimeout: 300, startedAt: start }),
    'timeout',
    'once a heavy line is seen the world timeout never fires; the old rule holds',
  );

  assert.equal(
    worldTimeoutMarker(300, { fps: 30, draws: 12 }),
    'harness: ended by --world-timeout after 300 s: the world never came (last frame line: 30 fps, 12 draws/frame)',
  );
  assert.equal(
    worldTimeoutMarker(600, null),
    'harness: ended by --world-timeout after 600 s: the world never came (no frame line)',
  );
});

// Shapes seen in a run, with made-up values: heavy frames drawing fine while the recording threads wait hundreds
// of ms a frame for pipelines, and the frame line saying how many are being made.
const frameAt = (mmss, { fps = 3, draws = 5000, dropped = 0, making = undefined, waiting = 0 } = {}) =>
  `2026-11-20T10:${mmss}.000000Z  INFO victory_gpu::host: ${fps} fps, 333.0 ms/frame (0.0 preparing: copies 0.0 ms 0.0 MB, textures 0.0 ms 0.0 MB; 0.0 on the GPU; 0.0 copying back 0.0 MB), ${draws} draws/frame, dropped ${dropped} without a host pipeline${making === undefined ? '' : ` (${making} being made)`} and 0 indirect; heap buffers 0.0 passes 0.0 ms; textures picked by index: 0 uploaded so far (0 MB), ${waiting} waiting, 0 passed over unmapped;`;
const waitAt = (mmss, ms) =>
  `2026-11-20T10:${mmss}.000000Z  INFO victory_gpu::copystat: recording per frame (105 frames): 29 list calls 4.6 ms (all recording threads); ${ms} ms waiting for pipelines; dynamic slots: 3 tables asked, 0 refused, 0.0 ms searching`;

test('pipelines being made or waited for hold the settle off, however good the frames look', () => {
  assert.equal(PIPELINE_QUIET_MS, 5);
  let state = initialState();
  for (const line of [frameAt('00:00', { making: 654 }), frameAt('00:05', { making: 654 }), frameAt('00:10', { making: 654 })]) {
    state = observe(state, line);
  }
  assert.equal(state.consecutive, 0, 'three good frames while 654 pipelines are being made are not a settle');
  assert.equal(state.settledAt, null);

  for (const line of [waitAt('00:12', 630), waitAt('00:18', 600), waitAt('00:24', 550)]) state = observe(state, line);
  state = observe(state, frameAt('00:26', { making: 0 }));
  assert.equal(state.settledAt, null, 'a quiet frame is not enough while the wait samples say compiling');
  assert.equal(state.pipelineRun, 0);

  for (const line of [waitAt('00:30', 2.0), frameAt('00:31', { making: 0 }), waitAt('00:36', 1.2), waitAt('00:42', 0.4)]) {
    state = observe(state, line);
  }
  assert.equal(state.pipelineRun, 3);
  assert.equal(state.metPipelinesAt, Date.parse('2026-11-20T10:00:42.000000Z'));
  for (const line of [frameAt('00:43', { making: 0 }), frameAt('00:48', { making: 0 }), frameAt('00:53', { making: 0 })]) {
    state = observe(state, line);
  }
  assert.equal(state.settledAt, Date.parse('2026-11-20T10:00:53.000000Z'));
  assert.equal(
    settledMarker(53, state.settledWaiting, state.settledFps, state.settledDetail),
    'harness: settled at 53 s (waiting 0, 3 fps); waited: frame rate from 0 s, pipelines from 42 s (3 samples under 5.0 ms, none being made), textures not gated (0 waiting), frame rate steady (spread 0%, 25% allowed)',
  );
});

test('a build that prints neither pipeline line settles as it always did, and the marker says not checked', () => {
  let state = initialState();
  for (const line of [frameAt('10:00'), frameAt('10:05'), frameAt('10:10')]) state = observe(state, line);
  assert.equal(state.settledAt, Date.parse('2026-11-20T10:10:10.000000Z'));
  assert.match(state.settledDetail, /pipeline quiet not checked/);
  assert.match(
    settledMarker(10, state.settledWaiting, state.settledFps, state.settledDetail),
    /^harness: settled at 10 s \(waiting 0, 3 fps\); waited: frame rate from 0 s, pipeline quiet not checked, textures not gated/,
  );
});

test('an unsettled run names the condition still unmet, once a minute', () => {
  assert.equal(UNSETTLED_NOTE_MS, 60_000);
  let state = initialState();
  state = observe(state, frameAt('01:00', { making: 654 }));
  state = observe(state, waitAt('01:02', 340));
  const why = unmetText(state, {});
  assert.match(why, /0 of 3 quiet pipeline samples \(last 340\.0 ms, under 5\.0 needed\)/);
  assert.match(why, /654 pipelines being made/);
  assert.match(
    unsettledMarker(64, why),
    /^harness: not settled after 64 s: 0 of 3 quiet pipeline samples \(last 340\.0 ms, under 5\.0 needed\); 654 pipelines being made$/,
  );

  for (const line of [waitAt('02:00', 0.2), waitAt('03:00', 0.1), waitAt('04:00', 0.5), frameAt('04:05', { making: 0 })]) {
    state = observe(state, line);
  }
  assert.match(unmetText(state, {}), /1 of 3 settled frame lines so far/, 'quiet now, but the frame run has restarted');

  // A build that never prints the pipeline lines says so, and the frame rate is the only thing it names.
  let plain = observe(initialState(), frameAt('10:00', { fps: 0.5 }));
  assert.match(unmetText(plain, {}), /frame rate 0\.5 fps under 1\.5/);
  assert.doesNotMatch(unmetText(plain, {}), /pipeline/);
});

test('a streaming scene whose frame rate swings is not settled; three steady lines are (audit: no stability test)', async () => {
  const sw = await import('./settled-watch.mjs');
  let state = initialState();
  // Heavy, quiet and over the floor, but swinging from 2 to 6 fps: the scene is still streaming in.
  for (const [mmss, fps] of [['20:00', 2], ['20:05', 6], ['20:10', 2]]) state = observe(state, frameAt(mmss, { fps, making: 0 }));
  assert.equal(state.settledAt, null, 'three lines at 2, 6 and 2 fps are not a settled scene');
  assert.match(unmetText(state, {}), /frame rate not steady: spread \d+% over the last 3 settled lines \(25% allowed\)/);

  // The window slides: the first steady line still sits beside two swinging ones.
  for (const [mmss, fps] of [['20:15', 4.0], ['20:20', 4.1]]) state = observe(state, frameAt(mmss, { fps, making: 0 }));
  assert.equal(state.settledAt, null);
  state = observe(state, frameAt('20:25', { fps: 3.9, making: 0 }));
  assert.equal(state.settledAt, Date.parse('2026-11-20T10:20:25.000000Z'), 'three lines within 25% of their mean');
  assert.match(state.settledDetail, /, frame rate steady \(spread 5%, 25% allowed\)$/);
  assert.equal(sw.MAX_FPS_SPREAD, 0.25);

  // Off means the rule of before: the swinging lines settle.
  let off = initialState();
  for (const [mmss, fps] of [['20:00', 2], ['20:05', 6], ['20:10', 2]]) {
    off = observe(off, frameAt(mmss, { fps, making: 0 }), { maxFpsSpread: null });
  }
  assert.equal(off.settledAt, Date.parse('2026-11-20T10:20:10.000000Z'));
});

test('a settle held by time does not fire before its wait has passed, and the unmet line says so', () => {
  let state = initialState();
  state = observe(state, frameAt('10:00', { making: 0 }), { settleAfterS: 150 });
  for (const line of [frameAt('10:10', { making: 0 }), frameAt('10:20', { making: 0 })]) {
    state = observe(state, line, { settleAfterS: 150 });
  }
  assert.equal(state.settledAt, null, '20 s after the first heavy line is before a 150 s wait');
  assert.equal(state.consecutive, 0);
  assert.match(unmetText(state, { settleAfterS: 150 }), /settle held until 150 s after the first heavy line/);

  for (const line of [frameAt('12:30', { making: 0 }), frameAt('12:40', { making: 0 }), frameAt('12:50', { making: 0 })]) {
    state = observe(state, line, { settleAfterS: 150 });
  }
  assert.equal(state.settledAt, Date.parse('2026-11-20T10:12:50.000000Z'));
});

// job/timing-truth's frame line keeps its beginning and ends with the window clause; the frame budget line follows it
// and is not a frame line. Made-up numbers, in the text victory-gpu host/stats.rs writes.
test('the window-shape frame line is read as before, and the frame budget line after it is not a frame line', () => {
  const line =
    '2026-11-26T05:02:11.250000Z  INFO victory_gpu::host: frame 4217, 4.5 fps, 220.9 ms/frame (12.5 preparing: copies 3.1 ms ' +
    "4.2 MB, textures hashed 5.3 ms 6.4 MB; 7.5 host waits (for the relay 1.5, for flights' fence 2.0, synchronous " +
    'submissions sent to retired 4.0); 8.6 copying back 9.7 MB merged into title memory (host time, fetches included), 0.4 MB ' +
    "fetched from copies in the card's memory), 5000 draws/frame, dropped 3 without a host pipeline (0 being made) and 7 " +
    'indirect; heap buffers 0.0 passes 0.0 ms; texture slots written so far: 1 placed, 1 of objects; textures picked by ' +
    'index: 1 uploaded so far (1 MB), 40 waiting, 0 passed over unmapped; host waits summed across threads 7.5 ms a frame, ' +
    'cpu 12%; means over frames 4213..4217 (5 frames in 1105 ms, p50 219.4, p95 231.1 ms/frame): every frame since the line ' +
    "before; the figures in the parentheses are summed across threads and overlap, they are not parts of the frame (the frame budget line's are)";
  assert.deepEqual(frameLine(line), { at: Date.parse('2026-11-26T05:02:11.250000Z'), fps: 4.5, draws: 5000, dropped: 3, making: 0, waiting: 40 });
  const budget =
    '2026-11-26T05:02:11.251000Z  INFO victory_gpu::host: frame budget, the presenting thread, frames 4213..4217 (5 frames): ' +
    "220.90 ms/frame = title 120.00 + recording 5.00 + unaccounted 95.90 (sum 220.90); of it on the processor: title 100.00 (title's " +
    "holds recording's); 310 scopes a frame; other threads per frame, overlapping, not parts of the frame: nothing measured";
  assert.equal(frameLine(budget), null);
});

test("the runtime's frame line (fixtures/runtime-lines.mjs) is read; its budget and segment lines are not frame lines", async () => {
  const { BUDGET_LINE, FRAME_LINE, SEGMENT_OPEN_LINE } = await import('./fixtures/runtime-lines.mjs');
  const prefix = '2026-11-26T10:00:00.000000Z  INFO victory_gpu::host: ';
  assert.deepEqual(frameLine(`${prefix}${FRAME_LINE}`), { at: Date.parse('2026-11-26T10:00:00.000000Z'), fps: 3, draws: 0, dropped: 0, making: 0, waiting: 0 });
  assert.equal(frameLine(`${prefix}${BUDGET_LINE}`), null);
  assert.equal(frameLine(`${prefix}${SEGMENT_OPEN_LINE}`), null);
});
