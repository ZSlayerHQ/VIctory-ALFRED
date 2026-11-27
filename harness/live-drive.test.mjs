import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  ONE_PASS,
  PlanError,
  RULES_TEXT,
  formatDriveReport,
  loadTimeline,
  loadTimelines,
  lockPath,
  normalizePlan,
  occurrencesOf,
  processAlive,
  reportRows,
  runPlan,
  statePath,
  timelinePath,
} from './live-drive.mjs';
import { parseAssignment } from './live.mjs';
import { LIVE_NAMES } from './live-names.mjs';
import { parseLiveLog } from './live-report.mjs';
import { DEFAULT_START, fakeRuntime, frameModel, makeClock, makeSim } from './live-fake-runtime.mjs';
import { isolatedEnv } from './test-isolation.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'live-drive.mjs');

const sink = { write() {} };
const fast = {
  holdScale: 0.02,
  markerTimeoutMs: 1000,
  confirmTimeoutMs: 1000,
  staleMs: 2000,
  stallMs: 3000,
  pollMs: 5,
  picture: false,
};

function tempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'live-drive-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

function stateOf(file) {
  let text = '';
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch {
    // A missing file means no assignments.
  }
  const state = {};
  for (const line of text.split(/\r?\n/)) {
    const assignment = parseAssignment(line);
    if (assignment && LIVE_NAMES.includes(assignment.name)) state[assignment.name] = assignment.value;
  }
  return state;
}

/// A run folder: the log, the control file and the stats file beside them, as a --live run lays them out.
function paths(t) {
  const dir = tempDir(t);
  return {
    dir,
    log: path.join(dir, 'run.log'),
    control: path.join(dir, 'live.txt'),
    stats: path.join(dir, 'stats.json'),
  };
}

/// The runtime as live.rs behaves (live-fake-runtime.mjs): only names present in the control file are applied, so a
/// removed name keeps its value; it writes the stats file a --live run has.
/// The fake runtime on a virtual clock, kept on `where` so drive() runs the plan on the same clock: nothing in these
/// tests waits on real time, so a slow or loaded machine gives the same results (the hosted runner's failure).
function fakeGame(t, where, options = {}) {
  const game = fakeRuntime(t, { log: where.log, control: where.control, statsAt: where.stats, ...options });
  where.clock = game.clock;
  return game;
}

function drive(plan, where, extra = {}) {
  return runPlan(plan, {
    logFile: where.log,
    controlFile: where.control,
    statsFile: where.stats,
    out: sink,
    err: sink,
    ...fast,
    ...(where.clock ? { now: where.clock.now, sleep: where.clock.sleep } : {}),
    ...extra,
  });
}

function planFor(steps, extra = {}) {
  return {
    baseline: { VICTORY_KEEP_TEXTURES: 'on' },
    steps,
    ...extra,
  };
}

const rowsOf = (where) => reportRows(parseLiveLog(fs.readFileSync(where.log, 'utf8')), loadTimeline(where.log));

test('a plan is normalized: values become strings, booleans become on/off', () => {
  const plan = normalizePlan({
    baseline: { VICTORY_KEEP_TEXTURES: true, VICTORY_TEXTURE_LOW: 60 },
    steps: [{ label: ' one ', set: { LIVE_SKIP_DRAW: '12,34' }, hold_s: 30 }],
  });
  assert.deepEqual(plan.baseline, { VICTORY_KEEP_TEXTURES: 'on', VICTORY_TEXTURE_LOW: '60' });
  assert.equal(plan.steps[0].label, 'one');
  // Corrected (was repeat 1): one pass never gives a direction, so a plan runs two unless it says otherwise.
  assert.equal(plan.repeat, 2);
  assert.equal(plan.settle_s, 0, 'no settling unless the plan asks for it');
  assert.equal(plan.steps[0].settle_s, 0);
});

test('the refusals: unknown names, a value the runtime refuses, over 40 steps, over 45 minutes, a bad hold, repeat or label', () => {
  const step = (label, hold = 30) => ({ label, set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: hold });
  const cases = [
    [
      { baseline: { NOT_A_SWITCH: '1' }, steps: [step('a')] },
      /baseline: 'NOT_A_SWITCH' is not a live switch/,
    ],
    [
      { baseline: {}, steps: [{ label: 'a', set: { lower: '1' }, hold_s: 30 }] },
      /not a switch name/,
    ],
    [
      { baseline: { VICTORY_TEXTURE_LOW: 'sixty' }, steps: [step('a')] },
      /VICTORY_TEXTURE_LOW must be a whole number/,
    ],
    [
      { baseline: {}, steps: Array.from({ length: 41 }, (_, i) => step(`s${i}`, 10)) },
      /41 steps; the limit is 40/,
    ],
    [
      { baseline: {}, steps: [step('slow', 300)], repeat: 5 },
      /the limit is 45 minutes/,
    ],
    [{ baseline: {}, steps: [step('short', 9)] }, /between 10 and 300 seconds/],
    [{ baseline: {}, steps: [step('long', 301)] }, /between 10 and 300 seconds/],
    [{ baseline: {}, steps: [step('a')], repeat: 0 }, /repeat must be a positive integer/],
    [{ baseline: {}, steps: [step('a'), step('a')] }, /duplicate label 'a'/],
    [
      { baseline: {}, steps: [{ label: 'two\nlines', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }] },
      /may not contain a newline/,
    ],
  ];
  for (const [plan, message] of cases) {
    assert.throws(() => normalizePlan(plan), message);
  }
});

test('the repeat reverses the list on the second pass and reverses back on the third', () => {
  const plan = normalizePlan({
    baseline: {},
    steps: [
      { label: 'a', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 },
      { label: 'b', set: { VICTORY_TEXTURE_HIGH: '85' }, hold_s: 10 },
    ],
    repeat: 3,
  });
  assert.deepEqual(
    occurrencesOf(plan).map((occurrence) => `${occurrence.label}:${occurrence.direction}`),
    ['a:forward', 'b:forward', 'b:reverse', 'a:reverse', 'a:forward', 'b:forward'],
  );
});

test('the runner brackets every step A, B, A, reverses the second pass, and ends at the baseline', async (t) => {
  const where = paths(t);
  const plan = planFor(
    [
      { label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 },
      { label: 'skip', set: { LIVE_SKIP_DRAW: '12,34' }, hold_s: 10 },
    ],
    { repeat: 2 },
  );
  const game = fakeGame(t, where, { baseline: plan.baseline });

  const result = await drive(plan, where);

  assert.deepEqual(
    result.occurrences.map((occurrence) => occurrence.label),
    ['low', 'skip', 'skip', 'low'],
  );
  assert.deepEqual(
    result.occurrences.map((occurrence) => occurrence.direction),
    ['forward', 'forward', 'reverse', 'reverse'],
  );
  // Corrected (was: the file holds only the baseline's own names): every touched name is written by value; and
  // again (the plan's start marks a segment): LIVE_MARK holds the plan's mark, one more than the start's 0.
  assert.deepEqual(
    stateOf(where.control),
    { VICTORY_KEEP_TEXTURES: 'on', VICTORY_TEXTURE_LOW: '70', LIVE_SKIP_DRAW: '', LIVE_MARK: '1' },
    'the baseline is restored, every switch the plan touched written by value',
  );
  assert.equal(game.state().VICTORY_TEXTURE_LOW, '70', 'the runtime is back at the start value');
  assert.equal(game.state().LIVE_SKIP_DRAW, '');

  for (let i = 0; i < result.occurrences.length; i += 1) {
    const occurrence = result.occurrences[i];
    assert.ok(occurrence.beforeSegment, `${occurrence.label}: a baseline segment before the step`);
    assert.ok(occurrence.stepSegment, `${occurrence.label}: a step segment`);
    assert.ok(occurrence.beforeSegment < occurrence.stepSegment, `${occurrence.label}: before is not the step`);
    if (i > 0) {
      assert.equal(
        result.occurrences[i - 1].afterSegment,
        occurrence.beforeSegment,
        'the A after one B is the A before the next',
      );
      assert.equal(result.occurrences[i - 1].afterArm, occurrence.beforeArm, 'and the same baseline arm');
    }
  }
  assert.equal(result.occurrences.at(-1).afterSegment, null, 'the final baseline closes only when the title exits');
  assert.equal(result.timeline.arms.length, 9, 'the first baseline, then a step and a baseline per occurrence');

  game.shutdown();
  const timeline = loadTimeline(where.log);
  assert.equal(timeline.version, 2);
  assert.equal(timeline.finishedAt != null, true);
  assert.equal(timeline.restored, true);
  assert.match(timeline.restoreNote, /confirmed by the stats file/);
  assert.equal(timeline.stopped, null);

  const rows = rowsOf(where);
  assert.deepEqual(rows.map((row) => row.label), ['low', 'skip']);
  for (const row of rows) {
    assert.equal(row.mean, 10);
    assert.equal(row.baseline, 20);
    assert.equal(row.difference, -10);
    assert.equal(row.percent, -50);
    assert.equal(row.spread, 0);
    assert.equal(row.verdict, 'faster');
    assert.deepEqual(row.reason, []);
  }
  assert.deepEqual(rows[0].inForce, { VICTORY_TEXTURE_LOW: ['60'] }, 'the step arms carry the values in force');
  const text = formatDriveReport(rows, timeline);
  assert.match(text, /\| low \| VICTORY_TEXTURE_LOW=60 \| VICTORY_TEXTURE_LOW=60 \| 10\.0 \|/);
  assert.ok(text.includes(RULES_TEXT), 'the rules are printed under the table');
});

test('both stats shapes measure the same arms; the p95 comes from the frame lists, or the log for the older shape', async (t) => {
  for (const shape of ['interval', 'sample']) {
    const where = paths(t);
    const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }]);
    const game = fakeGame(t, where, { baseline: plan.baseline, shape });
    const result = await drive(plan, where);
    game.shutdown();
    const step = result.timeline.arms[result.occurrences[0].stepArm];
    assert.equal(step.samples[0].shape, shape);
    if (shape === 'interval') {
      assert.ok(step.samples.every((sample) => sample.seq != null && sample.frames > 0 && sample.list.length > 0));
    } else {
      assert.ok(step.samples.every((sample) => sample.seq == null && sample.frames == null && sample.list == null));
    }
    const [row] = rowsOf(where);
    assert.equal(row.verdict, 'faster', shape);
    assert.equal(row.mean, 10, shape);
    // The fake's frame lists hold the state's time; its log segment's p95 is the mean plus 2.
    assert.equal(row.p95, shape === 'interval' ? 10 : 12, shape);
    assert.equal(row.p95Source, shape === 'interval' ? 'stats frame lists' : 'log segments', shape);
  }
});

test('objects written and never read are counted by the seq jump', async (t) => {
  const where = paths(t);
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }]);
  const game = fakeGame(t, where, { baseline: plan.baseline, seqStep: (tick, state) => (state.VICTORY_TEXTURE_LOW === '60' ? 3 : 1) });
  const result = await drive(plan, where);
  game.shutdown();
  const step = result.timeline.arms[result.occurrences[0].stepArm];
  assert.ok(step.excluded.missed >= 2, `missed counted (${step.excluded.missed})`);
  assert.match(formatDriveReport(rowsOf(where), loadTimeline(where.log)), /low: \d+ stats objects were written and never read/);
});

test('a step restored to its start value is written by value: the runtime keeps a removed name', async (t) => {
  // The audit's case: baseline {}, step FETCH_QUEUE=on. Removing the name restores nothing in the runtime.
  const where = paths(t);
  const plan = { baseline: {}, steps: [{ label: 'queue', set: { VICTORY_FETCH_QUEUE: 'on' }, hold_s: 10 }] };
  const game = fakeGame(t, where, { baseline: plan.baseline });

  const result = await drive(plan, where);
  assert.equal(game.state().VICTORY_FETCH_QUEUE, 'off', 'the queue is off again in the runtime');
  assert.deepEqual(stateOf(where.control), { VICTORY_FETCH_QUEUE: 'off', LIVE_MARK: '1' }, 'the start value is written, not unset');
  assert.equal(result.timeline.restored, true);
  assert.equal(result.timeline.start.VICTORY_FETCH_QUEUE, 'off', 'the start value came from the stats file');
  game.shutdown();
  const [row] = rowsOf(where);
  assert.equal(row.verdict, 'faster');
});

test('the runner refuses to start without a stats file, without its live map, or with a stale one', async (t) => {
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }]);
  const cases = [
    [{ stats: false }, /cannot read the stats file/],
    [{ statsLive: false }, /no whole object with frame, frame_ms and a live map/],
    // Corrected (was one wording): the interval shape is aged by its own clock, the older one by the file time.
    [{ statsAge: 60_000, statsEvery: 1_000_000 }, /last written 60(\.\d)? s ago by its own clock \(over 2 s\)/],
    [{ shape: 'sample', statsAge: 60_000, statsEvery: 1_000_000 }, /last written 60(\.\d)? s ago by its file time \(over 2 s\)/],
  ];
  for (const [options, message] of cases) {
    const where = paths(t);
    const game = fakeGame(t, where, { baseline: plan.baseline, ...options });
    await assert.rejects(drive(plan, where), (err) => err instanceof PlanError && message.test(err.message));
    assert.equal(fs.existsSync(where.control), false, `${message}: nothing is written`);
    game.stop();
  }
  const where = paths(t);
  fs.writeFileSync(where.log, '');
  await assert.rejects(
    runPlan(plan, { logFile: where.log, controlFile: where.control, out: sink, err: sink, ...fast }),
    /no stats file: pass --stats/,
  );
});

test('an arm starts only when the stats file shows the values asked', async (t) => {
  const where = paths(t);
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }]);
  // The marker comes at once; the stats file's live map shows the change only 80 ms later.
  const game = fakeGame(t, where, { baseline: plan.baseline, statsLagMs: 80 });
  const result = await drive(plan, where);
  game.shutdown();
  const step = result.timeline.arms[result.occurrences[0].stepArm];
  assert.ok(step.samples.length >= 3, 'the step arm has samples');
  for (const sample of step.samples) assert.equal(sample.live.VICTORY_TEXTURE_LOW, '60');
  const after = result.timeline.arms[result.occurrences[0].afterArm];
  for (const sample of after.samples) assert.equal(sample.live.VICTORY_TEXTURE_LOW, '70');
  assert.equal(rowsOf(where)[0].verdict, 'faster');
});

test('a value the runtime applies differently stops the run; the marker name alone is not a confirmation', async (t) => {
  // The validator's case 8: the plan asks 5, the runtime holds 30.
  const where = paths(t);
  const plan = planFor([{ label: 'low-5', set: { VICTORY_TEXTURE_LOW: '5' }, hold_s: 10 }]);
  fakeGame(t, where, { baseline: plan.baseline, applyAs: { VICTORY_TEXTURE_LOW: (value) => (value === '5' ? '30' : value) } });
  await assert.rejects(
    drive(plan, where),
    /stats file shows VICTORY_TEXTURE_LOW=30 \(asked 5\); the game did not apply what was asked/,
  );
  const timeline = loadTimeline(where.log);
  assert.equal(timeline.restored, true, 'the baseline (70) is applied and confirmed');
  const [row] = rowsOf(where);
  assert.equal(row.verdict, 'unreliable');
  assert.match(row.reason.join(' '), /the step was not measured/);
});

test('a baseline that cannot be confirmed after the plan is said, not called restored', async (t) => {
  const where = paths(t);
  const plan = { baseline: {}, steps: [{ label: 'keep off', set: { VICTORY_KEEP_TEXTURES: 'off' }, hold_s: 10 }] };
  // The runtime applies the step and then stops reading the control file; frames and stats go on. Corrected (was 1):
  // the plan's mark is the first change it applies, the step the second.
  fakeGame(t, where, { baseline: plan.baseline, ignoreAfterChanges: 2 });
  await assert.rejects(drive(plan, where, { markerTimeoutMs: 200 }), /waiting for the game to log VICTORY_KEEP_TEXTURES/);
  const timeline = loadTimeline(where.log);
  assert.equal(timeline.restored, false);
  assert.match(timeline.restoreNote, /shows VICTORY_KEEP_TEXTURES=off \(asked on\)/);
  assert.match(formatDriveReport(rowsOf(where), timeline), /the baseline is NOT confirmed after the plan/);
});

test('a switch flipped by hand mid-arm makes its row unreliable', async (t) => {
  // The validator's case 4b: TEXTURE_LOW set by hand during the step's hold, which slows every frame.
  const where = paths(t);
  const plan = { baseline: {}, steps: [{ label: 'queue', set: { VICTORY_FETCH_QUEUE: 'on' }, hold_s: 10 }] };
  const game = fakeGame(t, where, {
    meanOf: (state) => (state.VICTORY_TEXTURE_LOW === '40' ? 150 : state.VICTORY_FETCH_QUEUE === 'on' ? 80 : 100),
  });
  const out = {
    write(text) {
      // Corrected (was step 1/1): the plan runs two passes by default.
      if (/step 1\/2 'queue': confirmed .* holding/.test(text)) game.clock.after(60, () => game.handSet('VICTORY_TEXTURE_LOW', '40'));
    },
  };
  const result = await drive(plan, where, { out });
  game.shutdown();
  assert.deepEqual(result.timeline.othersMoved, [{ name: 'VICTORY_TEXTURE_LOW', start: '70', now: '40' }]);
  const [row] = rowsOf(where);
  assert.equal(row.verdict, 'unreliable');
  assert.match(row.reason.join(' '), /step 'queue'\): VICTORY_TEXTURE_LOW changed mid-arm \(70 -> 40\); set by hand\?/);
  assert.match(row.reason.join(' '), /baseline\): VICTORY_TEXTURE_LOW was 40 in force, not the 70 asked/);
});

test('the first baseline arm starts at the plan, not at the last change before it', async (t) => {
  // The validator's case 3c: the baseline is already in force, and the stretch before the plan ran at 110 ms.
  const where = paths(t);
  let planStarted = false;
  const game = fakeGame(t, where, {
    meanOf: (state) => (!planStarted ? 110 : state.VICTORY_FETCH_QUEUE === 'on' ? 80 : 100),
  });
  await game.clock.sleep(400);
  planStarted = true;
  const plan = { baseline: { VICTORY_FETCH_QUEUE: 'off' }, steps: [{ label: 'fq-on', set: { VICTORY_FETCH_QUEUE: 'on' }, hold_s: 10 }] };
  await drive(plan, where);
  game.shutdown();
  const [row] = rowsOf(where);
  assert.equal(row.baseline, 100, 'no frame from before the plan is in a baseline arm');
  assert.equal(row.mean, 80);
  assert.equal(row.difference, -20);
  assert.equal(row.verdict, 'faster');
});

test('stalled, stale and unparsable samples are counted and left out, and a short arm gives too few samples', async (t) => {
  const where = paths(t);
  const plan = { baseline: {}, steps: [{ label: 'queue', set: { VICTORY_FETCH_QUEUE: 'on' }, hold_s: 10 }] };
  let onTicks = 0;
  const stepTick = (state) => {
    if (state.VICTORY_FETCH_QUEUE !== 'on') {
      onTicks = 0;
      return 0;
    }
    return onTicks;
  };
  // The older shape: it rewrote its object while the frame stood still. The interval shape writes none then.
  const game = fakeGame(t, where, {
    shape: 'sample',
    meanOf: (state) => (state.VICTORY_FETCH_QUEUE === 'on' ? 80 : 100),
    // In the step: 3 good writes, then 6 whose frame stands still, then 6 torn ones, then none at all.
    // statsPaused is asked first on every write, so it keeps the count.
    statsPaused: (tick, state) => {
      if (state.VICTORY_FETCH_QUEUE === 'on') onTicks += 1;
      return stepTick(state) > 15;
    },
    statsFrozen: (tick, state) => stepTick(state) > 3 && stepTick(state) <= 9,
    statsGarbage: (tick, state) => stepTick(state) > 9 && stepTick(state) <= 15,
  });
  const result = await drive(plan, where, { holdScale: 0.05, staleMs: 60 });
  game.shutdown();
  const step = result.timeline.arms[result.occurrences[0].stepArm];
  assert.ok(step.excluded.stalled > 0, `stalled counted (${JSON.stringify(step.excluded)})`);
  assert.ok(step.excluded.unparsable > 0, 'unparsable counted');
  assert.ok(step.excluded.stale > 0, 'stale counted');
  assert.ok(step.samples.length < 3, `only the good samples are kept (${step.samples.length})`);
  const rows = rowsOf(where);
  assert.equal(rows[0].verdict, 'too few samples');
  assert.equal(rows[0].mean, null, 'no mean is given');
  assert.match(rows[0].reason.join(' '), /too few samples \(\d good over \d+ frames; 3 over 30 needed\)/);
  const text = formatDriveReport(rows, loadTimeline(where.log));
  assert.match(text, /\| queue \| VICTORY_FETCH_QUEUE=on \| VICTORY_FETCH_QUEUE=on \| n\/a \|/);
  assert.match(text, /samples left out: \d+ whose frame did not advance, \d+ stale \(no rewrite for 15 s, or an older seq\), \d+ unparsable/);
});

test('a small effect is no clear change, not faster', async (t) => {
  // The audit's case: a 0.5% change was called faster. The minimum effect is 3% of the baseline here.
  const where = paths(t);
  // The baseline names TEXTURE_LOW, so even a runner that restores by removal brings it back: only the verdict rule
  // is under test.
  const plan = {
    baseline: { VICTORY_KEEP_TEXTURES: 'on', VICTORY_TEXTURE_LOW: '70' },
    steps: [{ label: 'tiny', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }],
  };
  const game = fakeGame(t, where, { baseline: plan.baseline, baselineMean: 20, stepMean: 19.9 });
  await drive(plan, where);
  game.shutdown();
  const [row] = rowsOf(where);
  assert.equal(row.difference, -0.1);
  assert.equal(row.verdict, 'no clear change');
  assert.match(row.reason.join(' '), /the 0\.1 ms difference is not over the 0\.6 ms needed/);
});

test('a stop restores the baseline by value when the log stops growing, and says it could not confirm it', async (t) => {
  const where = paths(t);
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }]);
  fakeGame(t, where, { baseline: plan.baseline, stopAfterChanges: 1 });

  await assert.rejects(drive(plan, where, { stallMs: 200 }), /the run log has not grown for 200 ms; the game ended or hung/);
  // Corrected (was: the file holds only the baseline's names, and the timeline says restored).
  assert.deepEqual(stateOf(where.control), { VICTORY_KEEP_TEXTURES: 'on', VICTORY_TEXTURE_LOW: '70', LIVE_MARK: '1' });
  const timeline = loadTimeline(where.log);
  assert.match(timeline.stopped, /has not grown/);
  assert.equal(timeline.restored, false, 'a game that stopped cannot confirm the restore');
  assert.match(timeline.restoreNote, /no new stats sample within 1 s/);
});

test('a marker that never appears stops the run, restores the baseline and keeps the file comments', async (t) => {
  const where = paths(t);
  fs.writeFileSync(where.control, '# keep me\nVICTORY_KEEP_TEXTURES=on # the baseline note\n');
  const plan = {
    baseline: { VICTORY_KEEP_TEXTURES: 'on' },
    steps: [{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }],
  };
  fakeGame(t, where, { baseline: plan.baseline, answer: false });

  // Corrected (was: the step's VICTORY_TEXTURE_LOW, the first change the plan made): the plan's start writes its mark,
  // a change the runtime always logs, so a runtime that does not read the file is found at the first write.
  await assert.rejects(
    drive(plan, where, { markerTimeoutMs: 200 }),
    /timed out after 200 ms waiting for the game to log LIVE_MARK; the game is not reading the control file/,
  );
  // Corrected (was: the file is byte-for-byte as before): the comments stay, and the start value is written.
  assert.equal(
    fs.readFileSync(where.control, 'utf8'),
    '# keep me\nVICTORY_KEEP_TEXTURES=on # the baseline note\nVICTORY_TEXTURE_LOW=70\nLIVE_MARK=1\n',
  );
  const timeline = loadTimeline(where.log);
  assert.equal(timeline.restored, true);
  assert.match(timeline.stopped, /not reading the control file/);
});

test('a device-removed line stops the run and restores the baseline', async (t) => {
  const where = paths(t);
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }]);
  fakeGame(t, where, {
    baseline: plan.baseline,
    failureLine: 'DXGI_ERROR_DEVICE_REMOVED: /made/up/private/title/file was open',
  });

  let caught = null;
  await assert.rejects(drive(plan, where), (err) => {
    caught = err;
    return /the log carries 'DXGI_ERROR'/.test(err.message);
  });
  assert.equal(caught.message.includes('/made/up/private'), false, 'the raw log line is not echoed');
  assert.deepEqual(stateOf(where.control), { VICTORY_KEEP_TEXTURES: 'on', VICTORY_TEXTURE_LOW: '70', LIVE_MARK: '1' });
  assert.match(loadTimeline(where.log).stopped, /DXGI_ERROR/);
});

test('a step mean over five times the first baseline stops the run and restores the baseline', async (t) => {
  const where = paths(t);
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }]);
  fakeGame(t, where, { baseline: plan.baseline, stepMean: 200 });

  await assert.rejects(drive(plan, where), /over five times the first baseline's 20\.0 ms/);
  assert.deepEqual(stateOf(where.control), { VICTORY_KEEP_TEXTURES: 'on', VICTORY_TEXTURE_LOW: '70', LIVE_MARK: '1' });
  const timeline = loadTimeline(where.log);
  assert.match(timeline.stopped, /five times/);
  assert.equal(timeline.restored, true);
});

test('an interrupt during a hold restores the baseline', async (t) => {
  const where = paths(t);
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }]);
  fakeGame(t, where, { baseline: plan.baseline });
  const controller = new AbortController();
  where.clock.after(120, () => controller.abort());

  await assert.rejects(
    drive(plan, where, { signal: controller.signal }),
    (err) => err.exitCode === 130 && /interrupted/.test(err.message),
  );
  assert.deepEqual(stateOf(where.control), { VICTORY_KEEP_TEXTURES: 'on', VICTORY_TEXTURE_LOW: '70', LIVE_MARK: '1' });
  assert.equal(loadTimeline(where.log).restored, true);
});

test('a device-removed line already in the log stops before anything is written', async (t) => {
  const where = paths(t);
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }]);
  fakeGame(t, where, { baseline: plan.baseline, prelude: 'an earlier line: DXGI_ERROR_DEVICE_HUNG\n' });

  await assert.rejects(drive(plan, where), /the log carries 'DXGI_ERROR'/);
  // Corrected (was: the baseline is in the file): nothing was changed, so nothing is written to a dead game.
  assert.equal(fs.existsSync(where.control), false);
  assert.equal(loadTimeline(where.log).restoreNote, 'nothing was written');
});

test('a baseline already in force needs no marker, and the first step still measures', async (t) => {
  const where = paths(t);
  fs.writeFileSync(where.control, 'VICTORY_KEEP_TEXTURES=on\n');
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }]);
  const game = fakeGame(t, where, { baseline: plan.baseline });

  const result = await drive(plan, where);
  assert.equal(result.occurrences[0].stepSegment != null, true);
  game.shutdown();
  assert.equal(rowsOf(where)[0].verdict, 'faster');
});

test('a control file the runtime would ignore is refused before anything is written', async (t) => {
  const where = paths(t);
  fs.writeFileSync(where.log, 'fake title log\n');
  fs.writeFileSync(where.control, '# a comment\nnot an assignment\n');
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }]);

  await assert.rejects(drive(plan, where), /control file line 2 is not NAME=value/);
  assert.equal(fs.readFileSync(where.control, 'utf8'), '# a comment\nnot an assignment\n');
});

// ----------------------------------------------------------------------------------------------------------------
// the report on canned sidecars

/// A canned arm: `n` samples at `mean` ms, 20 frames apart, all with `live` in force (the target unless given).
function arm(index, kind, mean, { n = 5, target = { VICTORY_TEXTURE_LOW: kind === 'step' ? '60' : '70' }, live = null } = {}) {
  return {
    index,
    kind,
    label: null,
    target,
    startFrame: index * 1000,
    samples: Array.from({ length: n }, (_, i) => ({
      frame: index * 1000 + 20 * (i + 1),
      frame_ms: mean,
      fps: 1000 / mean,
      live: live ? live(i) : { ...target },
    })),
    excluded: { stalled: 0, stale: 0, unparsable: 0 },
  };
}

/// A version 2 sidecar for passes of [before, step, after] means; adjacent passes do not share arms here.
function canned(label, passes, { set = { VICTORY_TEXTURE_LOW: '60' }, armOptions = {} } = {}) {
  const arms = [];
  const occurrences = passes.map(([before, step, after], i) => {
    const base = arms.length;
    arms.push(arm(base, 'baseline', before, armOptions.before?.(i) ?? {}));
    arms.push(arm(base + 1, 'step', step, armOptions.step?.(i) ?? {}));
    if (after != null) arms.push(arm(base + 2, 'baseline', after, armOptions.after?.(i) ?? {}));
    return {
      label,
      set,
      pass: i + 1,
      direction: i % 2 ? 'reverse' : 'forward',
      beforeArm: base,
      stepArm: base + 1,
      afterArm: after != null ? base + 2 : null,
    };
  });
  return { version: 2, touched: Object.keys(set), start: { VICTORY_TEXTURE_LOW: '70' }, arms, occurrences, restored: true };
}

test('the report gives faster, slower and no clear change from the arms', () => {
  // Corrected (was one pass each): one pass gives no direction, so these are two passes.
  const [faster] = reportRows([], canned('fast', [[20, 10, 20], [20, 10, 20]]));
  assert.equal(faster.median, 10);
  assert.equal(faster.mean, 10);
  assert.equal(faster.baseline, 20);
  assert.equal(faster.difference, -10);
  assert.equal(faster.percent, -50);
  assert.equal(faster.verdict, 'faster');
  assert.deepEqual(faster.reason, []);

  const [slower] = reportRows([], canned('slow', [[20, 30, 20], [20, 30, 20]]));
  assert.equal(slower.difference, 10);
  assert.equal(slower.percent, 50);
  assert.equal(slower.verdict, 'slower');

  // Corrected (was 'no difference' by the old baseline-spread rule): the difference of 1 ms is under the 4 ms
  // needed, twice the 2 ms spread of the baseline arms.
  const [same] = reportRows([], canned('same', [[20, 21.5, 21], [21, 22.5, 22]]));
  assert.equal(same.mean, 22);
  assert.equal(same.baseline, 21);
  assert.equal(same.difference, 1);
  assert.equal(same.spread, 2);
  assert.equal(same.needed, 4);
  assert.equal(same.verdict, 'no clear change');
});

test('passes that disagree in sign give no clear change, with each pass shown', () => {
  // The validator's case 10: a direction came out of passes of -14.7 and +45.5 ms.
  const timeline = canned('copy-back pair', [[253.5, 244.2, 264.4], [254.7, 304.9, 264.0]]);
  const [row] = reportRows([], timeline);
  assert.deepEqual(row.passes, [-14.7, 45.5]);
  assert.equal(row.verdict, 'no clear change');
  assert.match(row.reason.join(' '), /the passes disagree in sign \(-14\.7 ms, \+45\.5 ms\)/);
  assert.match(formatDriveReport([row], timeline), /\| -14\.7, \+45\.5 \| no clear change \|/);
});

test('the report calls a drifted bracket no clear change when the baselines differ by more than the effect', () => {
  // Corrected (was 'unreliable' with a drift note): a drift is noise, so the effect must beat twice the spread.
  // Corrected (was one pass, [20, 24, 30]): one pass gives no verdict, so the drift runs back in a second pass.
  const [row] = reportRows([], canned('drift', [[20, 24, 30], [30, 24, 20]]));
  assert.equal(row.difference, -1);
  assert.equal(row.spread, 10);
  assert.equal(row.verdict, 'no clear change');
  assert.match(row.reason.join(' '), /the 1\.0 ms difference is not over the 20\.0 ms needed/);
});

test('samples that disagree with the target make the row unreliable and say what was in force', () => {
  const timeline = canned('low', [[20, 10, 20]], {
    armOptions: { step: () => ({ live: (i) => ({ VICTORY_TEXTURE_LOW: i < 2 ? '60' : '30' }) }) },
  });
  const [row] = reportRows([], timeline);
  assert.equal(row.verdict, 'unreliable');
  assert.deepEqual(row.inForce, { VICTORY_TEXTURE_LOW: ['60', '30'] });
  assert.match(row.reason.join(' '), /VICTORY_TEXTURE_LOW changed mid-arm \(60 -> 30\)/);
  const steady = canned('low', [[20, 10, 20]], {
    armOptions: { step: () => ({ live: () => ({ VICTORY_TEXTURE_LOW: '30' }) }) },
  });
  assert.match(reportRows([], steady)[0].reason.join(' '), /VICTORY_TEXTURE_LOW was 30 in force, not the 60 asked \(5 of 5 samples\)/);
  assert.match(formatDriveReport(reportRows([], steady), steady), /\| low \| VICTORY_TEXTURE_LOW=60 \| VICTORY_TEXTURE_LOW=30 \|/);
});

test('an arm with too few samples gives too few samples and no mean', () => {
  const [row] = reportRows([], canned('short', [[20, 10, 20]], { armOptions: { step: () => ({ n: 2 }) } }));
  assert.equal(row.verdict, 'too few samples');
  assert.equal(row.mean, null);
  assert.equal(row.difference, null);
  assert.match(row.reason.join(' '), /too few samples \(2 good over 40 frames; 3 over 30 needed\)/);
});

test('a missing closing baseline is unreliable and named, and the table shows the verdict words', () => {
  const timeline = canned('open', [[20, 10, null]]);
  const rows = reportRows([], timeline);
  assert.equal(rows[0].verdict, 'unreliable');
  assert.match(rows[0].reason.join(' '), /the baseline after the step was not measured/);
  const text = formatDriveReport(rows, timeline);
  // Corrected (was the old header): the table names the values in force, what was needed and every pass.
  // Corrected (was mean (ms), baseline (ms), needed (ms)): the figures are arm medians, and needed is in ms and %.
  assert.match(
    text,
    /\| label \| set to \| in force \(step arms\) \| step median \(ms\) \| p95 \(ms\) \| baseline median \(ms\) \| difference \| needed \| passes \(ms\) \| verdict \|/,
  );
  assert.match(text, /unreliable/);
});

test('an older sidecar or none gives unreliable rows that say why', () => {
  const seg = (segment, mean) =>
    `live segment ${segment}: frames ${segment * 300}..${segment * 300 + 229}, mean ${mean.toFixed(1)}, p50 ${mean.toFixed(1)}, ` +
    `p95 ${(mean + 30).toFixed(1)}, skipped 0 draws and 0 dispatches, in force: VICTORY_TEXTURE_LOW=70`;
  const segments = parseLiveLog([seg(14, 253.5), seg(15, 244.2), seg(16, 264.4)].join('\n'));
  const v1 = {
    version: 1,
    occurrences: [{ label: 'copy-back pair', pass: 1, set: {}, beforeSegment: 14, stepSegment: 15, afterSegment: 16 }],
  };
  const [old] = reportRows(segments, v1);
  assert.equal(old.verdict, 'unreliable');
  assert.equal(old.mean, null);
  assert.match(old.reason[0], /older sidecar or none.*restored the baseline by removing names/);
  const withChange = parseLiveLog(
    [seg(1, 20), 'live: VICTORY_TEXTURE_LOW 70 -> 60 at frame 530', seg(2, 10)].join('\n'),
  );
  const [none] = reportRows(withChange, null);
  assert.equal(none.label, 'VICTORY_TEXTURE_LOW=60');
  assert.equal(none.verdict, 'unreliable');
});

test('report --log --json reads the sidecar beside the log', (t) => {
  const dir = tempDir(t);
  const log = path.join(dir, 'run.log');
  fs.writeFileSync(log, 'fake title log\n');
  // Corrected (was one pass): one pass gives no verdict.
  fs.writeFileSync(timelinePath(log), JSON.stringify(canned('low', [[20, 10, 20], [20, 10, 20]])));
  const result = spawnSync(process.execPath, [script, 'report', '--log', log, '--json'], {
    encoding: 'utf8',
    env: isolatedEnv(),
  });
  assert.equal(result.status, 0, result.stderr);
  const rows = JSON.parse(result.stdout);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].label, 'low');
  assert.equal(rows[0].verdict, 'faster');
  const text = spawnSync(process.execPath, [script, 'report', '--log', log], {
    encoding: 'utf8',
    env: isolatedEnv(),
  });
  assert.equal(text.status, 0, text.stderr);
  assert.match(text.stdout, /# live-drive report/);
  assert.match(text.stdout, /\| low \| VICTORY_TEXTURE_LOW=60 \| VICTORY_TEXTURE_LOW=60 \| 10\.0 \|/);
});

function cli(args, extra = {}) {
  const env = isolatedEnv(extra);
  if (!('VICTORY_LIVE' in extra)) delete env.VICTORY_LIVE;
  if (!('VICTORY_LIVE_STATS' in extra)) delete env.VICTORY_LIVE_STATS;
  return spawnSync(process.execPath, [script, ...args], { encoding: 'utf8', env });
}

test('the command line refuses a missing plan, log, control file or stats file', (t) => {
  const dir = tempDir(t);
  const planFile = path.join(dir, 'plan.json');
  const logFile = path.join(dir, 'run.log');
  fs.writeFileSync(planFile, JSON.stringify(planFor([{ label: 'a', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }])));
  fs.writeFileSync(logFile, '');

  assert.equal(cli([]).status, 2);
  const noPlan = cli(['run', '--log', logFile, '--file', path.join(dir, 'live.txt')]);
  assert.equal(noPlan.status, 2);
  assert.match(noPlan.stderr, /run needs a plan file/);
  const noLog = cli(['run', planFile, '--file', path.join(dir, 'live.txt')]);
  assert.equal(noLog.status, 2);
  assert.match(noLog.stderr, /run needs --log/);
  const noControl = cli(['run', planFile, '--log', logFile]);
  assert.equal(noControl.status, 2);
  assert.match(noControl.stderr, /no control file/);
  const noStats = cli(['run', planFile, '--log', logFile, '--file', path.join(dir, 'live.txt')]);
  assert.equal(noStats.status, 2);
  assert.match(noStats.stderr, /no stats file: pass --stats/);
  assert.equal(fs.existsSync(path.join(dir, 'live.txt')), false, 'a refusal touches nothing');
});

test('the command line refuses an unknown live name before it writes the control file', (t) => {
  const dir = tempDir(t);
  const planFile = path.join(dir, 'plan.json');
  const logFile = path.join(dir, 'run.log');
  const control = path.join(dir, 'live.txt');
  fs.writeFileSync(logFile, '');
  fs.writeFileSync(
    planFile,
    JSON.stringify({ baseline: {}, steps: [{ label: 'bad', set: { NOT_A_SWITCH: '1' }, hold_s: 10 }] }),
  );
  const result = cli(['run', planFile, '--log', logFile, '--file', control]);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /'NOT_A_SWITCH' is not a live switch/);
  assert.equal(fs.existsSync(control), false, 'a refusal touches nothing');
});

test('the command line refuses a plan that is not JSON and a report without a log', (t) => {
  const dir = tempDir(t);
  const planFile = path.join(dir, 'plan.json');
  const logFile = path.join(dir, 'run.log');
  fs.writeFileSync(planFile, '{not json');
  fs.writeFileSync(logFile, '');
  const badPlan = cli(['run', planFile, '--log', logFile, '--file', path.join(dir, 'live.txt')]);
  assert.equal(badPlan.status, 2);
  assert.match(badPlan.stderr, /cannot read the plan/);
  const noLog = cli(['report']);
  assert.equal(noLog.status, 2);
  assert.match(noLog.stderr, /report needs --log/);
  const missingLog = cli(['report', '--log', path.join(dir, 'missing.log')]);
  assert.equal(missingLog.status, 1);
  assert.match(missingLog.stderr, /cannot read the run log/);
});

// ----------------------------------------------------------------------------------------------------------------
// the round-2 check's cases, on the simulated title (live-fake-runtime.mjs makeSim: the runtime's contract on a
// virtual clock; generated frames only)

const X = 'VICTORY_FETCH_QUEUE';
const xOn = (values) => values[X] === 'on';
const directional = (verdict) => verdict === 'faster' || verdict === 'slower';
// repeat is given, so the runner's step lines read 'step 1/2' whatever the default.
const xPlan = (hold = 30, extra = {}) => ({
  baseline: { [X]: 'off' },
  steps: [{ label: 'x-on', set: { [X]: 'on' }, hold_s: hold }],
  repeat: 2,
  ...extra,
});

/// Runs a plan through the real runner on a simulated title. `killAt`: the runner is dropped at that time with no
/// handler run (a process killed outright). Resolves { stopped, error } or { killed }, with what the runner said.
async function driveSim(sim, clock, plan, { killAt = null, killAfter = null, onSay = null, ...extra } = {}) {
  const runner = clock.runner({ killAt });
  const said = [];
  const out = {
    write(text) {
      said.push(text);
      onSay?.(text);
      // killAfter: {pattern, ms}: killed that long after the runner says a matching line.
      if (killAfter && runner.killAt == null && killAfter.pattern.test(text)) runner.killAt = sim.t + killAfter.ms;
    },
  };
  const run = runPlan(plan, {
    logFile: sim.log,
    controlFile: sim.control,
    statsFile: sim.stats,
    picture: false,
    out,
    err: out,
    now: runner.now,
    sleep: runner.sleep,
    ...extra,
  }).then(
    () => ({ stopped: null }),
    (error) => ({ stopped: error.message, error }),
  );
  const result = await Promise.race([run, runner.killed.then(() => ({ killed: true }))]);
  return { ...result, said: said.join('') };
}

const simRows = (sim) => reportRows(parseLiveLog(fs.readFileSync(sim.log, 'utf8')), loadTimeline(sim.log));

/// One seed: a fresh simulated title in its own folder, 20 s of play, then the plan. The older shape is polled once a
/// second (it writes one sample in about six), the interval shape at the runner's default so no object is missed.
async function simTrial(dir, { seed, shape, effect = 0, hold = 30, repeat, noise = 'real', extra = {} }) {
  const sim = makeSim({
    dir: path.join(dir, `${shape}-${seed}`),
    shape,
    seed,
    frameMs: frameModel({ noise, effect: (c) => (xOn(c.values) ? effect : 0) }),
  });
  sim.advanceTo(20_000);
  const result = await driveSim(sim, makeClock(sim), xPlan(hold, { repeat, ...extra }), {
    ...(shape === 'sample' ? { pollMs: 1000 } : {}),
  });
  const [row] = simRows(sim);
  fs.rmSync(path.dirname(sim.log), { recursive: true, force: true });
  return { ...result, row };
}

async function verdictsOver(dir, seeds, options) {
  const out = [];
  for (const seed of seeds) out.push((await simTrial(dir, { seed, ...options })).row);
  return out;
}

const seedsFrom = (first, count, step = 7) => Array.from({ length: count }, (_, i) => first + i * step);

test('one pass never gives a direction: a switch with no effect over the real log\'s noise', async (t) => {
  // The check: one pass (then the default), NO effect, real-log noise: 'slower' in 12 of 60 seeds (+3% to +104%).
  // Today's rule on these seeds gave a direction in 5 of 20 (the older shape) and 4 of 20 (the interval shape).
  const dir = tempDir(t);
  for (const [shape, count] of [['sample', 20], ['interval', 10]]) {
    const rows = await verdictsOver(dir, seedsFrom(1000, count), { shape, repeat: 1 });
    assert.deepEqual(
      rows.filter((row) => directional(row.verdict)).map((row) => `${row.verdict} ${row.percent}%`),
      [],
      `${shape}: one pass gave a direction`,
    );
    assert.ok(rows.every((row) => row.verdict === ONE_PASS), `${shape}: ${rows.map((row) => row.verdict).join(', ')}`);
    assert.ok(rows.every((row) => row.median != null && row.difference != null && row.needed != null), 'its figures are given');
    assert.match(rows[0].reason.join(' '), /one pass gives no direction/);
  }
});

test('two passes: no false direction for a switch with no effect, and a true -10% is found', async (t) => {
  const dir = tempDir(t);
  // No effect, 30 s arms, 2 passes: no direction in any seed (measured 0 of 60 in both shapes; the check 0 of 120).
  for (const [shape, count] of [['sample', 20], ['interval', 10]]) {
    const rows = await verdictsOver(dir, seedsFrom(1000, count), { shape, repeat: 2 });
    assert.deepEqual(rows.filter((row) => directional(row.verdict)).map((row) => row.percent), [], `${shape}: a false direction`);
  }
  // A true -10%, the interval shape (every frame in some object), 30 s arms, 2 passes: arm medians of the joined
  // frame lists found it in 58 of 60 seeds (1000 + 7i; 3 passes: 56 of 60) and in 14 of these 15; means of the same
  // objects (the old rule) found 3 of 20.
  const interval = await verdictsOver(dir, seedsFrom(1000, 15), { shape: 'interval', effect: -0.1, repeat: 2 });
  const found = interval.filter((row) => row.verdict === 'faster');
  assert.ok(found.length >= 13, `interval -10%: found ${found.length} of 15 (${interval.map((row) => row.verdict).join(', ')})`);
  assert.equal(interval.filter((row) => row.verdict === 'slower').length, 0, 'never the wrong direction');
  for (const row of found) assert.ok(row.percent < -6 && row.percent > -14, `the size is near -10% (${row.percent}%)`);
  // The older shape (one 1 s sample in about six), 60 s arms, 3 passes, the check's own seeds (1099 + 7i): arm medians
  // found it in 28 of 60 and in 11 of these 20 (the check, the old rule on the same medians: 25 of 60; its means: 0 of
  // 120). At 30 s arms and 2 passes medians find it in 11 of 60, as the check saw: with the older shape a 10% effect
  // needs long arms, and its no clear change says only that the effect is under `needed`.
  const sample = await verdictsOver(dir, seedsFrom(1099, 20), { shape: 'sample', effect: -0.1, hold: 60, repeat: 3 });
  const sampleFound = sample.filter((row) => row.verdict === 'faster').length;
  assert.ok(sampleFound >= 8, `sample -10% at 60 s x 3: found ${sampleFound} of 20`);
  assert.equal(sample.filter((row) => row.verdict === 'slower').length, 0, 'never the wrong direction');
});

test('the report prints good samples and frames per arm, needed in ms and %, and what no clear change means', () => {
  const timeline = canned('low', [[20, 19.8, 20], [20, 19.9, 20]]);
  const [row] = reportRows([], timeline);
  assert.equal(row.verdict, 'no clear change');
  assert.equal(row.needed, 0.6);
  assert.equal(row.neededPercent, 3);
  assert.deepEqual(row.armCounts, ['pass 1 5 (100)/5 (100)/5 (100)', 'pass 2 5 (100)/5 (100)/5 (100)']);
  const text = formatDriveReport([row], timeline);
  assert.match(text, /\| 0\.6 ms \(3\.0%\) \|/);
  assert.match(text, /low: good samples \(frames\) per arm, before\/step\/after: pass 1 5 \(100\)\/5 \(100\)\/5 \(100\); pass 2/);
  assert.match(text, /no clear change is not evidence of no effect/);
});

test('a pause arrives as one object holding the gap: it is named and left out (the simulated title)', async (t) => {
  // The check: a 12 s pause in the step arm gave 'slower +347%' with no note. The switch has NO effect here. The arms
  // are 45 s: in the older shape (one sample in about six seconds) a 12 s pause in a 30 s arm leaves too few samples.
  const dir = tempDir(t);
  for (const shape of ['sample', 'interval']) {
    let pauseFrom = Infinity;
    const quiet = frameModel({ noise: 'low' });
    const sim = makeSim({
      dir: path.join(dir, shape),
      shape,
      seed: 81,
      frameMs: (c) => (c.t >= pauseFrom && c.t < pauseFrom + 12_000 ? { pause: 500 } : quiet(c)),
    });
    sim.advanceTo(20_000);
    const result = await driveSim(sim, makeClock(sim), xPlan(45), {
      onSay: (text) => {
        if (/step 1\/2 'x-on': confirmed/.test(text)) pauseFrom = sim.t + 8000;
      },
    });
    assert.equal(result.stopped, null, result.stopped);
    const [row] = simRows(sim);
    assert.notEqual(row.verdict, 'slower', `${shape}: a direction from a pause`);
    assert.notEqual(row.verdict, 'faster', `${shape}: a direction from a pause`);
    assert.match(
      row.reason.join(' '),
      new RegExp(
        `arm 1 \\(step 'x-on'\\): 1 ${shape === 'interval' ? 'frame' : 'stats object'} over 5 x the arm's median ` +
          `\\(\\d+\\.\\d ms\\) left out as a pause or hitch: \\d{4,5}\\.\\d ms \\(stats frame \\d+\\)`,
      ),
      `${shape}: ${row.reason.join(' | ')}`,
    );
    assert.equal(row.excluded.pauses, 1, shape);
    assert.ok(Math.abs(row.repeats[0].difference) < 0.05 * row.baseline, `${shape}: the paused pass's effect ${row.repeats[0].difference}`);
  }
});

test('the fake writes a pause as one object holding the gap; the report names it and leaves it out', async (t) => {
  const where = paths(t);
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }], { repeat: 2 });
  const game = fakeGame(t, where, { baseline: plan.baseline });
  const out = {
    write(text) {
      if (/step 1\/2 'low': confirmed .* holding/.test(text)) game.clock.after(40, () => game.pause(150));
    },
  };
  // 500 ms arms, so the object after the pause is read inside the step arm.
  const result = await drive(plan, where, { out, holdScale: 0.05 });
  game.shutdown();
  const step = result.timeline.arms[result.occurrences[0].stepArm];
  const holding = step.samples.find((sample) => sample.list.some((ms) => ms >= 100));
  assert.ok(holding, 'one object carries the gap');
  assert.ok(holding.frame_ms > 10 && holding.list.filter((ms) => ms >= 100).length === 1, JSON.stringify(holding.list));
  const [row] = rowsOf(where);
  assert.equal(row.verdict, 'faster');
  assert.equal(row.p95, 10, 'the gap is not in the p95');
  assert.equal(row.excluded.pauses, 1);
  assert.match(row.reason.join(' '), /arm 1 \(step 'low'\): 1 frame over 5 x the arm's median \(10\.0 ms\) left out as a pause or hitch: \d{3,4}\.\d ms/);
});

test('a runner killed mid-arm leaves a state file; the next plan refuses until the start values are back', async (t) => {
  // The check: killed 10 s into a step arm, restarted with baseline {}: the leftover step value became the start
  // value, the step changed nothing, and the report said 'no clear change' for a -10% switch.
  const dir = tempDir(t);
  const sim = makeSim({ dir, seed: 112, frameMs: frameModel({ noise: 'low', effect: (c) => (xOn(c.values) ? -0.1 : 0) }) });
  const clock = makeClock(sim);
  sim.advanceTo(20_000);
  const plan = { baseline: {}, steps: [{ label: 'x-on', set: { [X]: 'on' }, hold_s: 30 }], repeat: 2 };
  const dead = 999_999_991;
  const alive = (pid) => pid === process.pid;
  const first = await driveSim(sim, clock, plan, {
    killAfter: { pattern: /step 1\/2 'x-on': confirmed/, ms: 10_000 },
    pid: dead,
    processAlive: alive,
  });
  assert.equal(first.killed, true);
  assert.equal(sim.values[X], 'on', 'the title is left in the step state');
  const left = JSON.parse(fs.readFileSync(statePath(sim.control), 'utf8'));
  assert.equal(left.pid, dead);
  assert.deepEqual(left.start, { [X]: 'off' });
  assert.equal(path.basename(left.sidecar), path.basename(timelinePath(sim.log)));
  const controlBefore = fs.readFileSync(sim.control, 'utf8');

  sim.advanceTo(sim.t + 10_000);
  const second = await driveSim(sim, clock, plan, { processAlive: alive });
  assert.ok(second.error instanceof PlanError, second.stopped ?? 'the second plan ran');
  assert.match(second.stopped, /an unfinished plan \(process 999999991, started .*sidecar .*live-drive\.json\) was not finished/);
  assert.match(second.stopped, /VICTORY_FETCH_QUEUE is on, was off/);
  assert.match(second.stopped, /harness\/live\.sh set VICTORY_FETCH_QUEUE=off/);
  assert.equal(fs.readFileSync(sim.control, 'utf8'), controlBefore, 'a refusal writes nothing');
  assert.equal(loadTimelines(sim.log).length, 1, 'and claims no sidecar');

  sim.handSet({ [X]: 'off' });
  sim.advanceTo(sim.t + 8000);
  const third = await driveSim(sim, clock, plan, { processAlive: alive });
  assert.equal(third.stopped, null, third.stopped);
  assert.match(third.said, /the stats file shows its start values again, so it is cleared/);
  assert.equal(fs.existsSync(statePath(sim.control)), false, 'a confirmed baseline removes the state file');
  assert.equal(fs.existsSync(lockPath(sim.control)), false, 'and the lock');
  sim.shutdown();
  const [row] = simRows(sim);
  assert.equal(row.verdict, 'faster', row.reason.join(' | '));
});

test('one runner per control file: a second is refused while the first runs, a stale lock is taken over', async (t) => {
  const where = paths(t);
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 10 }]);
  fakeGame(t, where, { baseline: plan.baseline });
  const first = drive(plan, where);
  await assert.rejects(
    drive(plan, where),
    (err) => err instanceof PlanError && /another live-drive runner \(process \d+, started .*\) holds live\.txt\.live-drive\.lock/.test(err.message),
  );
  await first;
  assert.equal(fs.existsSync(lockPath(where.control)), false, 'the lock is released');
  assert.equal(fs.existsSync(statePath(where.control)), false, 'the state file is removed');

  const exited = spawnSync(process.execPath, ['-e', ''], { env: isolatedEnv() });
  assert.equal(processAlive(process.pid), true);
  assert.equal(processAlive(exited.pid), false, 'a process that exited is not alive');
  fs.writeFileSync(lockPath(where.control), JSON.stringify({ pid: exited.pid, startedAt: 'earlier' }));
  const said = [];
  await drive(plan, where, { out: { write: (text) => said.push(text) } });
  assert.match(said.join(''), new RegExp(`took over a stale runner lock \\(its process ${exited.pid} no longer exists\\)`));
  assert.equal(fs.existsSync(lockPath(where.control)), false);
});

test('a step that changes nothing is refused: by the plan, or by the start values', async (t) => {
  assert.throws(
    () => normalizePlan({ baseline: { [X]: 'off' }, steps: [{ label: 'same', set: { [X]: 'off' }, hold_s: 10 }] }),
    /step 1 \('same'\): its set equals the baseline/,
  );
  const where = paths(t);
  fakeGame(t, where);
  const plan = { baseline: {}, steps: [{ label: 'already', set: { [X]: 'off' }, hold_s: 10 }] };
  await assert.rejects(
    drive(plan, where),
    (err) => err instanceof PlanError && /step 1 \('already'\) sets VICTORY_FETCH_QUEUE=off, which the baseline already has \(from the start values\)/.test(err.message),
  );
  assert.equal(fs.existsSync(where.control), false, 'nothing is written');
  assert.equal(fs.existsSync(statePath(where.control)), false);
});

test('every plan on one log keeps its own sidecar, and report lists each plan or the one --plan names', async (t) => {
  // The check: a second plan on the same log replaced the first plan's sidecar.
  const where = paths(t);
  const game = fakeGame(t, where);
  await drive({ baseline: {}, steps: [{ label: 'queue', set: { [X]: 'on' }, hold_s: 10 }] }, where);
  await drive({ baseline: {}, steps: [{ label: 'pool', set: { VICTORY_UPLOAD_POOL: 'on' }, hold_s: 10 }] }, where);
  game.shutdown();
  const plans = loadTimelines(where.log);
  assert.deepEqual(plans.map((entry) => entry.n), [1, 2]);
  assert.deepEqual(plans.map((entry) => path.basename(entry.file)), ['run.log.live-drive.json', 'run.log.live-drive.2.json']);
  assert.deepEqual(plans.map((entry) => entry.timeline.occurrences[0].label), ['queue', 'pool']);
  assert.equal(loadTimeline(where.log).occurrences[0].label, 'pool', 'loadTimeline gives the newest plan');

  const all = cli(['report', '--log', where.log]);
  assert.equal(all.status, 0, all.stderr);
  assert.match(all.stdout, /## plan 1 of 2 \(run\.log\.live-drive\.json, started [^)]+\)\n# live-drive report[\s\S]*\| queue \|/);
  assert.match(all.stdout, /## plan 2 of 2 \(run\.log\.live-drive\.2\.json, started [^)]+\)\n# live-drive report[\s\S]*\| pool \|/);
  const one = cli(['report', '--log', where.log, '--plan', '1']);
  assert.match(one.stdout, /\| queue \|/);
  assert.doesNotMatch(one.stdout, /\| pool \|/);
  const json = JSON.parse(cli(['report', '--log', where.log, '--json']).stdout);
  assert.deepEqual(json.map((row) => `${row.plan}:${row.label}`), ['1:queue', '2:pool']);
  assert.equal(cli(['report', '--log', where.log, '--plan', '3']).status, 2);
});

test('a switch flipped by hand and back between two samples makes the arm unreliable (the simulated title)', async (t) => {
  // The check: set off by hand and back on within 3 s, between two stats samples of the older shape: 'faster', no
  // note. The samples' live maps never show it; the log's markers and the stats file's segment count do.
  const dir = tempDir(t);
  const events = [];
  events.push({
    at: 66_000,
    do: (s) => {
      const last = s.samples.at(-1).t;
      const from = Math.max(last + (s.t - last < 1500 ? 600 : 6270), s.t + 100);
      events.push({ at: from, do: (s2) => s2.handSet({ [X]: 'off' }) }, { at: from + 3000, do: (s2) => s2.handSet({ [X]: 'on' }) });
    },
  });
  const sim = makeSim({
    dir,
    shape: 'sample',
    seed: 52,
    events,
    frameMs: frameModel({ noise: 'low', effect: (c) => (xOn(c.values) ? -0.1 : 0) }),
  });
  sim.advanceTo(20_000);
  const result = await driveSim(sim, makeClock(sim), xPlan(30), { pollMs: 1000 });
  assert.equal(result.stopped, null, result.stopped);
  sim.shutdown();
  const flips = sim.markers.filter((marker) => marker.t >= 66_000 && marker.t < 66_000 + 15_000);
  assert.equal(flips.length >= 2, true, 'the hand flips applied');
  const [row] = simRows(sim);
  assert.equal(row.verdict, 'unreliable', row.reason.join(' | '));
  assert.match(
    row.reason.join(' '),
    new RegExp(`arm 1 \\(step 'x-on'\\): the log shows VICTORY_FETCH_QUEUE on -> off at frame ${flips[0].frame}, inside the arm`),
  );
});

test('a closed-segment count that moves within an arm makes it unreliable, even with no marker in the log', () => {
  const timeline = canned('low', [[20, 10, 20], [20, 10, 20]]);
  for (const arm of timeline.arms) {
    arm.startSegment = 4;
    for (const sample of arm.samples) sample.segment = 4;
  }
  timeline.arms[1].samples[3].segment = 6;
  timeline.arms[1].samples[4].segment = 6;
  const [row] = reportRows([], timeline);
  assert.equal(row.verdict, 'unreliable');
  assert.match(
    row.reason.join(' '),
    /arm 1 \(step 'low'\): the stats file's count of closed segments went 4 -> 6 inside the arm \(frames 1000\.\.1100\): a switch changed, and perhaps changed back, between two samples/,
  );
  // The log's markers count too, when only the log shows the change.
  const marked = canned('low', [[20, 10, 20], [20, 10, 20]]);
  marked.changes = [{ name: X, old: 'off', new: 'on', frame: 1060 }];
  assert.match(reportRows([], marked)[0].reason.join(' '), /the log shows VICTORY_FETCH_QUEUE off -> on at frame 1060, inside the arm \(frames 1000\.\.1100\)/);
});

test('settle_s leaves an arm\'s first seconds out as settling, counted and named', async (t) => {
  // The check: a true -10% that arrives 12 s after the switch applies (and leaves 12 s after it is undone) read -5.1%.
  assert.throws(() => normalizePlan(xPlan(30, { settle_s: -1 })), /settle_s must be a number of seconds, 0 or more/);
  assert.throws(() => normalizePlan(xPlan(20, { settle_s: 12 })), /settle_s is 12; it may be at most half of hold_s \(10 s\)/);
  assert.equal(normalizePlan({ ...xPlan(30, { settle_s: 5 }), steps: [{ ...xPlan(30).steps[0], settle_s: 12 }] }).steps[0].settle_s, 12);
  const lagged = (c) => {
    const since = c.since[X];
    if (xOn(c.values)) return since >= 12_000 ? -0.1 : 0;
    return since != null && since < 12_000 ? -0.1 : 0;
  };
  const dir = tempDir(t);
  const figures = {};
  for (const settle of [0, 12]) {
    const sim = makeSim({ dir: path.join(dir, String(settle)), seed: 31, frameMs: frameModel({ noise: 'low', effect: lagged }) });
    sim.advanceTo(20_000);
    const result = await driveSim(sim, makeClock(sim), xPlan(30, { settle_s: settle }));
    assert.equal(result.stopped, null, result.stopped);
    [figures[settle]] = simRows(sim);
  }
  assert.ok(figures[0].percent > -8, `without settling the lag hides part of the effect (${figures[0].percent}%)`);
  assert.ok(figures[12].percent < -9 && figures[12].percent > -11, `settled: ${figures[12].percent}%`);
  assert.equal(figures[12].verdict, 'faster');
  assert.ok(figures[12].excluded.settling >= 5, `counted (${figures[12].excluded.settling})`);
  assert.match(figures[12].reason.join(' '), /arm 1 \(step 'x-on'\): \d+ samples read in its first 12 s left out as settling \(settle_s\)/);
});

test('the fake runtime keeps a removed name, as live.rs does', async (t) => {
  // The test double's own contract, so no test can again pass on the wrong rule.
  const where = paths(t);
  fs.writeFileSync(where.control, 'VICTORY_FETCH_QUEUE=on\n');
  const game = fakeGame(t, where);
  await game.clock.sleep(50);
  assert.equal(game.state().VICTORY_FETCH_QUEUE, 'on');
  fs.writeFileSync(where.control, '');
  await game.clock.sleep(50);
  assert.equal(game.state().VICTORY_FETCH_QUEUE, 'on', 'a removed name keeps its value');
  assert.equal(DEFAULT_START.VICTORY_FETCH_QUEUE, 'off');
  const stats = JSON.parse(fs.readFileSync(where.stats, 'utf8'));
  assert.equal(stats.live.VICTORY_FETCH_QUEUE, 'on');
  assert.equal('VICTORY_LIVE_HUD' in stats.live, false, 'the HUD switch is left out while off');
});

test('the shell wrapper forwards to the module', () => {
  const result = spawnSync('bash', [path.join(here, 'live-drive.sh'), '--help'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /usage: live-drive\.sh run <plan\.json>/);
});

// The rules that read the runtime's own lines and objects (the check of 2026-11-26 found the first-shape readers
// let them never fire on a real run).

test("the plan's start writes a mark: the stretch before it is a segment of its own, and the first arm begins at the mark", async (t) => {
  const where = paths(t);
  const game = fakeGame(t, where, { meanOf: (state) => (state.VICTORY_FETCH_QUEUE === 'on' ? 80 : 100) });
  await game.clock.sleep(200);
  const plan = { baseline: {}, steps: [{ label: 'queue', set: { VICTORY_FETCH_QUEUE: 'on' }, hold_s: 10 }], repeat: 1 };
  await drive(plan, where);
  const segments = parseLiveLog(fs.readFileSync(where.log, 'utf8'));
  assert.deepEqual(segments[0].changes, [], 'the stretch before the plan');
  assert.equal(segments[0].ended, 'mark', 'is ended by the mark');
  assert.equal(segments[1].begun, 'mark', 'the first baseline arm begins at it');
  assert.equal(segments[1].transitionExcluded, 0, 'a mark changes nothing, so no frame is left out as transition');
  assert.deepEqual(segments[1].changes.map((change) => [change.name, change.new]), [['LIVE_MARK', '1']]);
  assert.equal(loadTimeline(where.log).mark, '1');
  assert.equal(game.state().LIVE_MARK, '1');
  // A second plan on the same run takes the next number.
  await drive(plan, where);
  assert.equal(game.state().LIVE_MARK, '2');
  game.shutdown();
});

test('a plan may not set LIVE_MARK: it is the runner\'s own', () => {
  assert.throws(() => normalizePlan({ baseline: { LIVE_MARK: '3' }, steps: [{ label: 'a', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 30 }] }), /LIVE_MARK is the runner's own/);
});

test("the samples keep the closed-segment count from segments_closed_total, and a count that moves inside an arm makes its row unreliable", async (t) => {
  const where = paths(t);
  const game = fakeGame(t, where, { meanOf: (state) => (state.VICTORY_FETCH_QUEUE === 'on' ? 80 : 100) });
  const plan = { baseline: {}, steps: [{ label: 'queue', set: { VICTORY_FETCH_QUEUE: 'on' }, hold_s: 10 }] };
  await drive(plan, where);
  game.shutdown();
  const timeline = loadTimeline(where.log);
  const samples = timeline.arms.flatMap((arm) => arm.samples);
  assert.ok(samples.length > 10);
  assert.ok(samples.every((sample) => Number.isFinite(sample.segments_closed)), 'every sample has the count');
  const segments = parseLiveLog(fs.readFileSync(where.log, 'utf8'));
  assert.equal(reportRows(segments, timeline)[0].verdict, 'faster', 'as run, nothing moved');
  // A switch flipped and flipped back between two samples: the log's markers lost, only the count moved.
  const step = timeline.arms[1];
  const half = Math.floor(step.samples.length / 2);
  for (const sample of step.samples.slice(half)) sample.segments_closed += 2;
  const [row] = reportRows(segments, timeline);
  assert.equal(row.verdict, 'unreliable');
  assert.ok(row.reason.some((reason) => /count of closed segments went \d+ -> \d+ inside the arm/.test(reason)), row.reason.join('; '));
});

test("the five-times rule holds the open segment, once it has counted enough frames, against the first closed baseline's", async (t) => {
  const where = paths(t);
  const plan = planFor([{ label: 'low', set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 60 }], { repeat: 1 });
  fakeGame(t, where, { baseline: plan.baseline, stepMean: 200 });
  await assert.rejects(drive(plan, where), /the open segment \d+ means 200\.0 ms, over five times the first baseline's 20\.0 ms/);
  const timeline = loadTimeline(where.log);
  assert.match(timeline.stopped, /five times/);
  const step = timeline.arms[1];
  assert.ok(step.endedMs - step.startedMs < 0.5 * 60 * 1000 * fast.holdScale, 'stopped early in the step, not when it closed');
});
