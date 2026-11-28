import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { isolatedEnv } from './test-isolation.mjs';
import { BUDGET_LINE, BUDGET_LINE_OTHERS } from './fixtures/runtime-lines.mjs';

// The query mark of a web address, joined in at run time so that no address literal here carries one.
const Q = '?';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'summarize-run.mjs');
const TS = '2026-11-25T00:00:00.000000Z  INFO victory_gpu::host: ';
const REASON = 'feeds or traces ray-tracing structures, which are not built; left out';

function frame({ fps = 30, ms = 33.4, draws = 5000, dropped = 0 } = {}) {
  return (
    `${TS}${fps} fps, ${ms} ms/frame (0.4 preparing: copies 0.2 ms 0.3 MB, textures 0.0 ms 0.0 MB; ` +
    `0.0 on the GPU; 0.0 copying back 0.0 MB), ${draws} draws/frame, dropped ${dropped} without a host pipeline ` +
    `(0 being made) and 0 indirect; heap buffers 0.0 passes 0.0 ms; texture slots written so far: 1 placed, ` +
    `1 of objects; textures picked by index: 1 uploaded so far (1 MB), 0 waiting, 0 passed over unmapped`
  );
}

function fixture(t, lines) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'summarize-run-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const file = path.join(dir, 'run.log');
  fs.writeFileSync(file, lines.join('\n') + '\n');
  return file;
}

test("tonight's new lines reach the summary's sections", (t) => {
  const log = fixture(t, [
    'harness: settled at 75 s (waiting 40, 2.4 fps)',
    frame({ ms: 300 }),
    `${TS}host capabilities: gpu_upload_heaps=1 enhanced_barriers=1 tiled_tier=1 binding_tier=1 dedicated_video_bytes=1 local_budget_bytes=1`,
    // From the specified format: no thread clock in a log yet.
    `${TS}card clock threads: frame=1.0 | "victory-copy" 100.0 1.0 30.0 | "quiet" 2.0 0.0 1.0 | other 5.0 1.0 2.5 | total 107.0 2.0 33.5`,
    frame({ ms: 320 }),
    `${TS}card clock threads: frame=2.0 | "victory-copy" 120.0 1.0 30.0 | "quiet" 2.0 0.0 1.0 | other 5.0 1.0 2.5 | total 127.0 2.0 33.5`,
    `${TS}table constants per frame (VICTORY_TABLE_CONSTANTS=1): 13 read in place, 2 copied (memory read in place 0, no copy or past it 0, off 16 bytes in its copy 2, in pages rewritten under a reader 0); 0.0 refreshes waited 0.0 ms for a submission reading constants in place`,
    // From the specified format: not yet in a log.
    `${TS}buffer copies per frame: 5 on the card from host copies (2.0 MB), 1 read from title memory on the CPU (1.0 MB: the switch off 0, off 4 bytes 1, no host copy holding an end 0, memory the card reads in place 0, overlapping ends 0, no copy pipeline 0), 4 of them settled the flights in the air (0.8 MB)`,
    `${TS}submissions sent ahead per frame: 3.7 in 1.5 settles (0.0 flights each time), 120.0 ms waiting for the card in them; settled for: title memory read 8.0 (by a buffer copy 5.0, a fill of upload memory 2.0, a texture 1.0); 0.0 waits for memory values met by a flight's bottom-of-pipe write, 1.4 fence signals set behind flights`,
    `${TS}of the waiting for the card in settles, per frame, by what settled: a synchronous submission or present 0.5 ms, the card had run them 1.0, a wait for a memory value 0.0, title memory changed under shader writes 2.0, title memory read 7.0, a step of ours 0.0, lists in the air 0.0, bottom-of-pipe writes 0.0, a blocking fence wait 0.0`,
    `${TS}dropped 10.0/frame without a host pipeline: ${REASON}`,
  ]);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /## Pipeline health/);
  assert.match(result.stdout, /- left out:\n {2}- 10\.0 a frame: feeds or traces ray-tracing structures/);
  assert.match(result.stdout, /## Card and CPU per frame/);
  assert.match(result.stdout, /- host capabilities: gpu_upload_heaps=1 /);
  assert.match(result.stdout, /- table constants: 13\.0 read in place, 2\.0 copied/);
  assert.match(result.stdout, /- buffer copies: 5\.0 on the card \(2\.0 MB\), 1\.0 on the CPU \(1\.0 MB\), 4\.0 settled flights \(0\.8 MB\)/);
  assert.match(result.stdout, / {2}- title memory read: 8\.0 a frame, 7\.0 ms\n {4}- by a buffer copy: 5\.0/);
  assert.match(result.stdout, /## Threads/);
  assert.match(result.stdout, /- victory-copy: 111\.0 ms \(30\.0%\)/);
  assert.match(result.stdout, /- total: 119\.0 ms \(33\.5%\)/);
});

test('--json carries the same aggregates', (t) => {
  const log = fixture(t, [
    'harness: settled at 75 s (waiting 40, 2.4 fps)',
    frame({ ms: 300 }),
    `${TS}table constants per frame (VICTORY_TABLE_CONSTANTS=1): 13 read in place, 2 copied (off 16 bytes in its copy 2); 0.0 refreshes waited 0.0 ms`,
    `${TS}dropped 10.0/frame without a host pipeline: ${REASON}`,
  ]);
  const result = spawnSync(process.execPath, [script, log, '--json'], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  const parsed = JSON.parse(result.stdout);
  assert.equal(parsed.speedLines.tableConstants.inPlace, 13);
  assert.equal(parsed.speedLines.leftOut[REASON], 10);
});

test('a log without the new lines prints none of the new sections', (t) => {
  const log = fixture(t, [frame({ ms: 300 })]);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.includes('## Card and CPU per frame'), false);
  assert.equal(result.stdout.includes('## Threads'), false);
});

test('the instrument lines reach Run health, and NOT CLEAN is said outright', (t) => {
  const clean = "2520 guards put on, 77 faults taken, 0 guards gone with no fault, 2 pages passed over as a kernel write or the platform layer held them, 0 not armed for their protection, 3 openings around our own kernel copies; the instrument's cost 0.038 ms a frame";
  const log = fixture(t, [
    frame({ ms: 300 }),
    `${TS}null magenta: frame=120 substituted=3 never_made=2 waiting=1 let_go=0 cannot_make=1 unmapped=1 stale=0 left_depth=1 left_integer=0 left_other=0 left_no_view=2 examples=0x10:waiting,0x20:unmapped`,
    `${TS}page shares: kernel 12%, platform 3%; health: ${clean}`,
    `${TS}NOT CLEAN: 4 guards went with no fault (2 unmapped, 2 lost); health: ${clean.replace('0 guards gone with no fault', '4 guards gone with no fault')}`,
  ]);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /## Run health/);
  assert.match(result.stdout, /- null magenta: 3\.0 substituted a frame \(worst 3\.0\), 2\.0 never made, 1\.0 waiting/);
  assert.match(result.stdout, /examples: 0x10:waiting, 0x20:unmapped/);
  assert.match(result.stdout, /- first touch: \*\*NOT CLEAN\*\* — 4\.0 guards went with no fault/);
  assert.match(result.stdout, /cost 0\.038 ms a frame/);
});

test('the summary opens with the comparable verdict, and --json carries it', (t) => {
  const quietStart = 'harness: load at start: cpu 4% over 5 s, free memory 42.1 GB of 63.8, commit 38.2 of 64.0 GB, builders 0';
  const quietEnd = 'harness: load at end: cpu 7% over 5 s, free memory 40.0 GB of 63.8, commit 39.0 of 64.0 GB, builders 0';
  const log = fixture(t, [quietStart, frame(), quietEnd, 'harness: victory exit=0']);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /^# Run summary — .*\ncomparable: yes\n/);

  const json = spawnSync(process.execPath, [script, log, '--json'], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(json.status, 0, json.stderr);
  const parsed = JSON.parse(json.stdout);
  assert.equal(parsed.comparable, 'yes');
  assert.equal(parsed.load.start.cpu, 4);
  assert.equal(parsed.load.end.builders, 0);
});

test('a busy start makes the summary not comparable, and thresholds can move', (t) => {
  const hotStart = 'harness: load at start: cpu 41% over 5 s, free memory 42.1 GB of 63.8, commit 38.2 of 64.0 GB, builders 0';
  const quietEnd = 'harness: load at end: cpu 7% over 5 s, free memory 40.0 GB of 63.8, commit 39.0 of 64.0 GB, builders 0';
  const log = fixture(t, [hotStart, frame(), quietEnd, 'harness: victory exit=0']);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /^# Run summary — .*\ncomparable: no \(cpu 41% is over 25% at start\)\n/);

  const moved = spawnSync(process.execPath, [script, log], {
    encoding: 'utf8',
    env: isolatedEnv({ HARNESS_MAX_CPU: '50' }),
  });
  assert.equal(moved.status, 0, moved.stderr);
  assert.match(moved.stdout, /^# Run summary — .*\ncomparable: yes\n/);
});

test('a log without the load lines says the verdict is unknown', (t) => {
  const log = fixture(t, [frame(), 'harness: victory exit=0']);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /^# Run summary — .*\ncomparable: unknown \(no load lines\)\n/);
  const json = JSON.parse(
    spawnSync(process.execPath, [script, log, '--json'], { encoding: 'utf8', env: isolatedEnv() }).stdout,
  );
  assert.equal(json.comparable, 'unknown (no load lines)');
});

// The frame line the runtime's format writes from job/honest-counters on, colour codes included.
const NEW_FRAME =
  '\x1b[2m2026-11-26T05:02:11.250000Z\x1b[0m \x1b[32m INFO\x1b[0m ' +
  '\x1b[2mvictory_gpu::host\x1b[0m\x1b[2m:\x1b[0m ' +
  'frame 18432, 27.4 fps, 36.5 ms/frame (0.4 preparing: copies 0.3 ms 1.2 MB, ' +
  'textures hashed 0.2 ms 0.1 MB; 1.7 on the GPU; 0.1 copying back 0.9 MB merged into ' +
  "title memory (host time, fetches included), 0.0 MB fetched from copies in the card's memory), " +
  '5000 draws/frame, dropped 3 without a host pipeline (1 being made) and 7 indirect';
const NEW_COPIES =
  '2026-11-26T05:02:11.800000Z  INFO victory_gpu::copystat: copies per frame: 1.0 submissions; ' +
  '2 title copies 3.0 MB; table fills 4.0 MB asked, 5.0 MB filled in 0.10 ms; ' +
  'descriptor-copy sync 0.25 ms (12 records); 6 refreshes 7.0 MB, 8 skipped 9.0 MB; ' +
  '10 runs 0.20 ms; 11 queries 0.30 ms, moves 0.40 ms, 12.0 MB unmapped; 13 page faults; ' +
  '14 watches 0.50 ms (1 failed 0.01 ms), 15.0 MB asked about (repeats counted), 16.0 MB unchanged, ' +
  '17 distinct pages since the last present; 18 fallbacks; 19 guard faults; 20 idle copies kept, ' +
  '21 let go by pressure; textures 22.0 MB watched and changed, 23.0 MB hashed, 24.0 MB of them changed';
const NEW_RECORDING =
  '2026-11-26T05:02:11.800000Z  INFO victory_gpu::copystat: recording per frame (105 frames): ' +
  '31 list calls 5.2 ms (all recording threads); 0.0 ms waiting for pipelines; dynamic slots: ' +
  '4 tables asked, 0 refused, 0.0 ms searching; notes 0.0 ms; shader-written ranges: 0 steps in ' +
  '0 submissions, snapshots 0.0 ms, 0.0 MB compared back; sparse reads 0.0 ms; ' +
  "42 guards in 0.7 ms (the fault handler's included); 0.0 MB copied whole (memory not watched); " +
  '0.0 frame waits sleeping 0.0 ms (all threads); ranges of shaders that cannot store to buffers: ' +
  '0 steps 0.0 MB left out';
const OBSERVERS =
  '2026-11-26T05:02:10.000000Z  INFO victory_gpu::host: observers on: the debug layer, ' +
  'GPU-based validation, DRED, GPU timing, the memory map walk, the counting allocator';
const SWITCHES =
  '2026-11-26T05:02:10.100000Z  INFO victory_gpu::switches: diagnostic switch set ' +
  'switch=VICTORY_D3D12_DEBUG value=1 on=true';

test('the first lines name the observers, and the new copystat fields reach the summary and --json', (t) => {
  const quietStart = 'harness: load at start: cpu 4% over 5 s, free memory 42.1 GB of 63.8, commit 38.2 of 64.0 GB, builders 0';
  const quietEnd = 'harness: load at end: cpu 7% over 5 s, free memory 40.0 GB of 63.8, commit 39.0 of 64.0 GB, builders 0';
  const log = fixture(t, [quietStart, OBSERVERS, SWITCHES, NEW_FRAME, NEW_COPIES, NEW_RECORDING, quietEnd, 'harness: victory exit=0']);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.match(
    result.stdout,
    /^# Run summary — .*\ncomparable: yes\nobservers on: the debug layer, GPU-based validation, DRED, GPU timing, the memory map walk, the counting allocator\n/,
  );
  assert.match(result.stdout, /switches set: VICTORY_D3D12_DEBUG=1 \(on\)/);
  assert.match(result.stdout, /last presented frame: 18432/);
  assert.match(result.stdout, /Last frame: frame 18432, 36\.5 ms/);
  assert.match(result.stdout, /textures hashed 0\.2 ms \/ 0\.1 MB/);
  assert.match(
    result.stdout,
    /Copy stats: descriptor-copy sync 0\.25 ms \(12 records\), 15\.0 MB asked about, 17 distinct pages since the last present, 42 guards in 0\.7 ms/,
  );

  const json = JSON.parse(
    spawnSync(process.execPath, [script, log, '--json'], { encoding: 'utf8', env: isolatedEnv() }).stdout,
  );
  assert.equal(json.lastFrame.frame, 18432);
  assert.equal(json.lastFrameLine.frame, 18432);
  assert.deepEqual(json.copyStats, {
    descriptorCopySyncMs: 0.25,
    descriptorCopyRecords: 12,
    distinctPages: 17,
    guardCount: 42,
    guardsMs: 0.7,
    askedAboutMb: 15,
  });
  assert.equal(json.descriptorCopySyncMs, 0.25);
  assert.equal(json.descriptorCopyRecords, 12);
  assert.equal(json.distinctPages, 17);
  assert.equal(json.guardCount, 42);
  assert.deepEqual(json.observers, [
    'the debug layer',
    'GPU-based validation',
    'DRED',
    'GPU timing',
    'the memory map walk',
    'the counting allocator',
  ]);
  assert.deepEqual(json.switches, [{ name: 'VICTORY_D3D12_DEBUG', value: '1', on: true }]);
});

test('an old-form log keeps its summary, with the observers line named absent', (t) => {
  const log = fixture(t, [frame(), 'harness: victory exit=0']);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /observers on: not in the log/);
  const json = JSON.parse(
    spawnSync(process.execPath, [script, log, '--json'], { encoding: 'utf8', env: isolatedEnv() }).stdout,
  );
  assert.equal(json.lastFrameLine.frame, null);
  assert.equal(json.copyStats, null);
  assert.equal(json.observers, null);
  assert.deepEqual(json.switches, []);
});

// The cost part's fourth figure is the host's waits (the audit of 2026-11-26), whichever label the build wrote.
const WAITS_FRAME =
  '2026-11-26T05:02:11.250000Z  INFO victory_gpu::host: frame 900, 27.4 fps, 36.5 ms/frame (0.4 preparing: ' +
  "copies 0.3 ms 1.2 MB, textures hashed 0.2 ms 0.1 MB; 7.5 host waits (for the relay 1.5, for flights' fence 2.0, " +
  'synchronous submissions sent to retired 4.0); 0.1 copying back 0.9 MB), 5000 draws/frame, dropped 3 without a ' +
  'host pipeline (1 being made) and 7 indirect';

test('the host waits print as host waits, never as GPU time, in the last frame and the worst frames', (t) => {
  const log = fixture(t, [WAITS_FRAME, 'harness: victory exit=0']);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  const lastFrame = result.stdout.split('\n').find((line) => line.startsWith('Last frame:'));
  assert.match(lastFrame, /host waits 7\.5 ms/);
  assert.match(result.stdout, /## Worst frames\n- \S+ 27\.4 fps \/ 36\.5 ms: copies 0\.3 ms 1\.2 MB, host waits 7\.5 ms,/);
  assert.doesNotMatch(result.stdout, /\bGPU\s+[\d.]/);
  const json = JSON.parse(
    spawnSync(process.execPath, [script, log, '--json'], { encoding: 'utf8', env: isolatedEnv() }).stdout,
  );
  assert.equal(json.costs.hostWaitsMs, 7.5);
  assert.equal(json.lastFrameLine.hostWaitsMs, 7.5);
  assert.equal('gpuMs' in json.costs, false);
  assert.equal('gpuMs' in json.lastFrameLine, false);
});

test('a field absent from the log prints n/a, never a dash or zero', (t) => {
  // A frame line whose cost part is in a wording the parser does not know: every cost is absent, not zero.
  const unknownCosts =
    '2026-11-26T05:02:11.250000Z  INFO victory_gpu::host: frame 900, 27.4 fps, 36.5 ms/frame (a wording from a ' +
    'later build), 5000 draws/frame, dropped 3 without a host pipeline (1 being made) and 7 indirect';
  const log = fixture(t, [unknownCosts, 'harness: victory exit=0']);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  const lastFrame = result.stdout.split('\n').find((line) => line.startsWith('Last frame:'));
  assert.match(
    lastFrame,
    /preparing n\/a \(copies n\/a ms \/ n\/a MB, textures hashed n\/a ms \/ n\/a MB\), host waits n\/a ms, readback n\/a ms \/ n\/a MB/,
  );
  const worst = result.stdout.split('## Worst frames\n')[1].split('\n')[0];
  assert.match(worst, /copies n\/a ms n\/a MB, host waits n\/a ms/);
  for (const line of [lastFrame, worst]) assert.doesNotMatch(line, /—\s*(ms|MB|\)|,)|\b0\.0 ms/);

  // No frame line at all: the frame-rate figures are absent, and the range of a log with no timestamps too.
  const empty = fixture(t, ['harness: victory exit=0']);
  const bare = spawnSync(process.execPath, [script, empty], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(bare.status, 0, bare.stderr);
  assert.match(bare.stdout, /^FPS: 0 samples, median n\/a, min n\/a, max n\/a, last n\/a$/m);
  assert.match(bare.stdout, /^Range: n\/a; last presented frame: n\/a$/m);
});

test('a run with no PresentMon CSV says PresentMon: not run', (t) => {
  const log = fixture(t, [WAITS_FRAME, 'harness: victory exit=0']);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /^PresentMon: not run$/m);
});

// Synthetic run text for the validator's probes of 2026-11-26 (made up here; no real log is copied).
const at = (base, seconds) => new Date(Date.parse(base) + seconds * 1000).toISOString().replace('Z', '000Z');
function numberedFrame(stamp, frameNumber, fps = 30) {
  return (
    `${stamp}  INFO victory_gpu::host: frame ${frameNumber}, ${fps} fps, 33.3 ms/frame (0.4 preparing: copies 0.3 ms ` +
    '1.2 MB, textures hashed 0.2 ms 0.1 MB; 7.5 on the GPU; 0.1 copying back 0.9 MB), 5000 draws/frame, dropped 0 ' +
    'without a host pipeline (0 being made) and 0 indirect'
  );
}
function synthRun(base, frameNumbers, { header = true } = {}) {
  return [
    ...(header ? ['harness build: abc1234 profile=diag console=series-x'] : []),
    ...frameNumbers.map((n, i) => numberedFrame(at(base, i * 5), n)),
    'harness: victory exit=0',
  ];
}

test('frame lines in a form the parser does not read are counted and warned, and the last frame is n/a', (t) => {
  // The fps field renamed: every frame line fails the parser, and only the power-of-two present line is left.
  const renamed = [0, 5, 10].map((s) => numberedFrame(at('2026-11-26T10:00:00Z', s), 100 + s).replace(' fps, ', ' FPS, '));
  const present =
    '2026-11-26T10:00:11.000000Z  INFO victory_gpu::queue: frame presented (since the last line: draws and ' +
    'dispatches, and those run on the host) frame=2048 draws=12 drawn=12 dispatches=0 dispatched=0';
  const log = fixture(t, [...renamed, present, 'harness: victory exit=0']);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.match(
    result.stdout,
    /^warning: 3 frame lines \(lines with ms\/frame\) are in a form this parser does not read; their figures are left out$/m,
  );
  assert.match(result.stdout, /last presented frame: n\/a$/m);
  const json = JSON.parse(
    spawnSync(process.execPath, [script, log, '--json'], { encoding: 'utf8', env: isolatedEnv() }).stdout,
  );
  assert.equal(json.unreadFrameLines, 3);
  assert.equal(json.lastFrame.frame, null);
});

// Only a line that begins as a frame line (after the log's prefix) and does not parse is an unread frame line. The
// copystat block's per-source lines and the frame budget line also hold "ms/frame"; they are not frame lines (the
// check of 2026-11-26 found a log of 690 such lines warned as unread, where the true count was 0). Made-up values.
test('copy-stat and budget lines that hold ms/frame are not counted as unread frame lines', (t) => {
  const copiedWhole = '  copied whole from 0x7f0000000000+0x40000: 2.0x/frame, 0.52 MB/frame, 0.13 ms/frame';
  const prefix = '2026-11-26T10:00:01.000000Z  INFO victory_gpu::copystat: ';
  const copiedFrom = `${prefix}  copied from 0x7f0000100000+0x1000 (memory 0x3): 1.5x/frame, 0.01 MB/frame, 0.02 ms/frame`;
  const budget = `${prefix}${BUDGET_LINE}`;
  const log = fixture(t, [numberedFrame(at('2026-11-26T10:00:00Z', 0), 100), copiedWhole, copiedFrom, budget, 'harness: victory exit=0']);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.doesNotMatch(result.stdout, /in a form this parser does not read/);
  const json = JSON.parse(spawnSync(process.execPath, [script, log, '--json'], { encoding: 'utf8', env: isolatedEnv() }).stdout);
  assert.equal(json.unreadFrameLines, 0);
});

test('a log that holds two runs is warned and only its last run is summarized, never the two added together', (t) => {
  const cases = {
    'a second run header': [...synthRun('2026-11-26T10:00:00Z', [100, 200, 300, 400, 500]), ...synthRun('2026-11-26T12:00:00Z', [100, 200, 300])],
    'the frame counter restarting': [
      ...synthRun('2026-11-26T10:00:00Z', [100, 200, 300, 400, 500]),
      ...synthRun('2026-11-26T12:00:00Z', [100, 200, 300], { header: false }),
    ],
  };
  for (const [reason, lines] of Object.entries(cases)) {
    const log = fixture(t, lines);
    const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
    assert.equal(result.status, 0, result.stderr);
    assert.match(
      result.stdout,
      new RegExp(`^warning: this log holds 2 runs \\(${reason} at line \\d+\\); this summary covers the last one only$`, 'm'),
      reason,
    );
    assert.match(result.stdout, /^Range: 10 s; last presented frame: 300$/m, reason);
    assert.match(result.stdout, /^FPS: 3 samples,/m, reason);
    const json = JSON.parse(
      spawnSync(process.execPath, [script, log, '--json'], { encoding: 'utf8', env: isolatedEnv() }).stdout,
    );
    assert.equal(json.runsInLog.count, 2, reason);
    assert.equal(json.durationSeconds, 10, reason);
  }
  // Timestamps going back with no header and no frame number between them: joined too.
  const back = fixture(t, [
    ...synthRun('2026-11-26T12:00:00Z', [100, 200, 300, 400]).slice(1).map((line) => line.replace(/frame \d+, /, '')),
    ...synthRun('2026-11-26T10:00:00Z', [100, 200, 300], { header: false }).map((line) => line.replace(/frame \d+, /, '')),
  ]);
  const result = spawnSync(process.execPath, [script, back], { encoding: 'utf8', env: isolatedEnv() });
  assert.match(result.stdout, /^warning: this log holds 2 runs \(the timestamps going back at line \d+\); this summary covers the last one only$/m);
  // One run: no warning.
  const one = fixture(t, synthRun('2026-11-26T10:00:00Z', [100, 200, 300]));
  assert.doesNotMatch(spawnSync(process.execPath, [script, one], { encoding: 'utf8', env: isolatedEnv() }).stdout, /^warning:/m);
});

test('a settle marker written without a texture count is read, and its count prints n/a', (t) => {
  const log = fixture(t, [
    numberedFrame('2026-11-26T10:00:00.000000Z', 100),
    'harness: settled at 12 s (waiting ?, 4.1 fps); waited: x',
    numberedFrame('2026-11-26T10:00:05.000000Z', 200),
    'harness: victory exit=0',
  ]);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /^Scene settled: 12 s after the first heavy line \(waiting n\/a, 4\.1 fps\)\.$/m);
});

// PresentMon is the judge of frame times (the owner, 2026-11-26): when it ran, its median and p95 come first, before
// any of the runtime's own figures.
test('when PresentMon ran, its median and p95 frame time print before the runtime figures', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'summarize-run-pm-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const capture = path.join(dir, 'capture');
  fs.mkdirSync(capture);
  fs.writeFileSync(
    path.join(capture, 'presentmon.csv'),
    'Application,ProcessID,TimeInMs,MsBetweenPresents,MsGPUBusy\r\n' +
      ['victory.exe,1,0,30,5', 'victory.exe,1,30,32,5', 'victory.exe,1,62,34,5', 'victory.exe,1,96,50,5'].join('\r\n') + '\r\n',
  );
  const log = path.join(dir, 'run.log');
  fs.writeFileSync(log, [`capture: ${capture}`, WAITS_FRAME, 'harness: victory exit=0'].join('\n') + '\n');
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  const lines = result.stdout.split('\n');
  const pm = lines.findIndex((line) => line.startsWith('PresentMon: '));
  assert.match(lines[pm], /^PresentMon: 4 frames — median 33\.0 ms, p95 50\.0 ms/);
  for (const runtime of ['Range: ', 'FPS: ', 'Last frame: ']) {
    const at = lines.findIndex((line) => line.startsWith(runtime));
    assert.ok(at > pm, `${runtime} comes after PresentMon's figures`);
  }
});

// The lines of job/timing-truth with made-up numbers, in the text victory-gpu host/stats.rs writes.
const windowFrame = ({ frame = 105, ms = 220.0, first = 101, frames = 5, wall = 1100, p50 = 219.4, p95 = 231.1, at = '2026-11-26T05:02:11.250000Z' } = {}) =>
  `${at}  INFO victory_gpu::host: frame ${frame}, ${(1000 / ms).toFixed(1)} fps, ${ms.toFixed(1)} ms/frame (12.5 preparing: copies 3.1 ms 4.2 MB, ` +
  "textures hashed 5.3 ms 6.4 MB; 7.5 host waits (for the relay 1.5, for flights' fence 2.0, synchronous submissions sent to " +
  'retired 4.0); 8.6 copying back 9.7 MB merged into title memory (host time, fetches included), 0.4 MB fetched from copies ' +
  "in the card's memory), 5000 draws/frame, dropped 3 without a host pipeline (1 being made) and 7 indirect; heap buffers " +
  '0.0 passes 0.0 ms; host waits summed across threads 7.5 ms a frame, cpu 12%; ' +
  `means over frames ${first}..${frame} (${frames} frames in ${wall} ms, p50 ${p50}, p95 ${p95} ms/frame): every frame since ` +
  "the line before; the figures in the parentheses are summed across threads and overlap, they are not parts of the frame (the frame budget line's are)";
// The budget line as the runtime's tests pin it (fixtures/runtime-lines.mjs BUDGET_LINE_OTHERS), its frames and phases
// replaced where a test needs its own; every other character stays the runtime's.
const budgetLine = ({ first = 101, last = 105, frame = '200.00', phases, sum = '200.00' } = {}) => {
  let text = BUDGET_LINE_OTHERS.replace('frames 1230..1234 (2 frames): 200.00 ms/frame', `frames ${first}..${last} (2 frames): ${frame} ms/frame`).replace(
    '(sum 200.00)',
    `(sum ${sum})`,
  );
  if (phases) text = text.replace(/= .+? \(sum /, `= ${phases} (sum `);
  return `2026-11-26T05:02:11.251000Z  INFO victory_gpu::host: ${text}`;
};
const summarize = (t, lines, args = []) => {
  const result = spawnSync(process.execPath, [script, fixture(t, lines), ...args], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  return result.stdout;
};

test('the summary says which frame-line shape it read, and joins the windows weighted by their frames', (t) => {
  const lines = [
    windowFrame({ frame: 105, ms: 200, first: 101, frames: 5, wall: 1000, p50: 199, p95: 210 }),
    windowFrame({ frame: 120, ms: 300, first: 106, frames: 15, wall: 4500, p50: 290, p95: 400, at: '2026-11-26T05:02:16.250000Z' }),
    'harness: victory exit=0',
  ];
  const text = summarize(t, lines);
  assert.match(text, /^Frame lines: 2, all in the window shape \(each a mean over every frame since the line before\)$/m);
  assert.match(text, /^Frame windows: frames 101\.\.120, 20 frames in 5500 ms: 275\.0 ms\/frame \(3\.6 fps\), weighted by frames; the last line's p50 290\.0, p95 400\.0 ms$/m);
  const json = JSON.parse(summarize(t, lines, ['--json']));
  assert.equal(json.frameWindow.meanMs, 275);
  assert.deepEqual(json.frameLines, { window: 2, older: 0 });
});

test("the frame budget prints every phase it has with unaccounted, the processor's share and the other threads", (t) => {
  const text = summarize(t, [windowFrame(), budgetLine(), 'harness: victory exit=0']);
  assert.match(text, /^## Frame budget \(the presenting thread\)$/m);
  assert.match(text, /^- 1 line, frames 101\.\.105 \(2 frames\): 200\.00 ms\/frame = title 100\.00 \+ unaccounted 20\.00 \+ relay_wait 80\.00 \(sum 200\.00\)$/m);
  assert.match(text, /^- identity: the phases sum to the frame on all 1 line within rounding \(tolerance: half a unit of the last printed place per figure added, 0\.020 ms at most here\)$/m);
  assert.match(text, /^- on the processor: the title and its calls together 0\.00, unaccounted 10\.00, relay_wait 40\.00$/m);
  assert.match(text, /^- 30 scopes a frame, 0 calls from the title a frame, the largest stretch of title between two 0\.00 ms$/m);
  assert.match(text, /^- other threads per frame, overlapping, not parts of the frame: relay_land 60\.00 \(cpu 15\.00\)$/m);
  assert.doesNotMatch(text, /^warning: the frame budget/m);
});

test("a budget whose phases do not sum to its frame is warned in a warning: line", (t) => {
  const broken = budgetLine({ frame: '161.60', sum: '161.60', phases: 'title 100.00 + list_calls 0.00 + unaccounted 61.60 + settle 110.90' });
  const text = summarize(t, [windowFrame(), budgetLine(), windowFrame({ frame: 110, first: 106 }), broken.replace('101..105', '106..110'), 'harness: victory exit=0']);
  assert.match(
    text,
    /^warning: the frame budget's phases do not sum to the frame on 1 of 2 lines \(worst at frames 106\.\.110: phases 272\.50 ms against sum 161\.60 and frame 161\.60 ms; tolerance 0\.025 ms\)$/m,
  );
});

test("the runtime's budget line with the title's calls is summarized with its calls and largest stretch", (t) => {
  const text = summarize(t, [windowFrame(), `2026-11-26T05:02:11.251000Z  INFO victory_gpu::host: ${BUDGET_LINE}`, 'harness: victory exit=0']);
  assert.match(text, /= title 60\.00 \+ list_calls 30\.00 \+ unaccounted 100\.00 \+ object_calls 6\.00 \+ hle_calls 4\.00 \(sum 200\.00\)$/m);
  assert.match(text, /^- on the processor: the title and its calls together 90\.00, unaccounted 50\.00$/m);
  assert.match(text, /^- 30 scopes a frame, 2000 calls from the title a frame, the largest stretch of title between two 3\.50 ms$/m);
  assert.doesNotMatch(text, /^warning: the frame budget/m);
});

test('a budget line naming a phase the runtime does not is warned: its shape moved', (t) => {
  const renamed = budgetLine({ phases: 'title 100.00 + recording 20.00 + relay_wait 80.00' });
  const text = summarize(t, [windowFrame(), renamed, 'harness: victory exit=0']);
  assert.match(text, /^warning: the frame budget names 1 phase the runtime's list does not \(recording\)/m);
});

test('an older log reads n/a for the frame windows and the budget, and names the shape it read', (t) => {
  const text = summarize(t, [numberedFrame('2026-11-26T10:00:00.000000Z', 100), 'harness: victory exit=0']);
  assert.match(text, /^Frame lines: 1, all in the older shape \(a sample, no window clause\)$/m);
  assert.match(text, /^Frame windows: n\/a \(no frame line has the window clause\)$/m);
  assert.match(text, /^Frame budget: n\/a \(no frame budget line: a build without diagnostics, or from before job\/timing-truth\)$/m);
  assert.doesNotMatch(text, /## Frame budget/);
});

// Made-up addresses on reserved names: the summary never prints a line that holds one (web-address.mjs), in text or
// in --json, and says how many it withheld. The check of 2026-11-26 found them in the failures, the slowest commands
// and the refused pipelines, unmasked.
test('a log line holding a web address never reaches the summary, in text or --json; the count is said', (t) => {
  const log = fixture(t, [
    numberedFrame('2026-11-26T10:00:00.000000Z', 100),
    '2026-11-26T10:00:01.000000Z ERROR victory_gpu::host: device removed, see HTTPS://example.com/help' + Q + 'id=7 for the reason',
    '2026-11-26T10:00:02.000000Z ERROR victory_gpu::host: device removed (DXGI_ERROR_DEVICE_HUNG) at D:/captures/x',
    'harness: victory exit=1',
  ]);
  const text = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(text.status, 0, text.stderr);
  assert.doesNotMatch(text.stdout, /example\.com/i);
  assert.match(text.stdout, /DXGI_ERROR_DEVICE_HUNG/, 'a failure line without an address is still shown');
  assert.match(text.stdout, /^- 1 line withheld \(they hold a web address\)$/m);
  const json = spawnSync(process.execPath, [script, log, '--json'], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(json.status, 0, json.stderr);
  assert.doesNotMatch(json.stdout, /example\.com/i);
  const parsed = JSON.parse(json.stdout);
  assert.ok(parsed.withheldLines >= 1);
  assert.ok(parsed.failures.some((line) => /DXGI_ERROR_DEVICE_HUNG/.test(line)));
});


// Their protocol (--protocol team-270): their two windows after the settle, from PresentMon's own frames of the
// game's process, and their results template filled where the log has a source.
function protocolFixture(t, { rows, logLines, extraEnv = [] }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'summarize-run-team-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const capture = path.join(dir, 'capture');
  fs.mkdirSync(capture);
  fs.writeFileSync(path.join(capture, 'presentmon.csv'), rows);
  const log = path.join(dir, 'run.log');
  fs.writeFileSync(
    log,
    ['harness build: test profile=diag console=series-x branch=HEAD', `capture: ${capture}`, 'harness env: VICTORY_LOG=info', ...extraEnv, ...logLines].join('\n') + '\n',
  );
  return log;
}

const med = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const pct = (xs, p) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(Math.max(0, Math.ceil(p * s.length) - 1), s.length - 1)];
};
function windowNumbers(values) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const slow = [...values].sort((a, b) => b - a).slice(0, Math.max(1, Math.round(values.length * 0.01)));
  return {
    frames: values.length,
    fps: (1000 / mean).toFixed(1),
    median: med(values).toFixed(1),
    p90: pct(values, 0.9).toFixed(1),
    p99: pct(values, 0.99).toFixed(1),
    low1: (1000 / (slow.reduce((a, b) => a + b, 0) / slow.length)).toFixed(1),
    worst: Math.max(...values).toFixed(1),
    gpu: ((100 * 50 * values.length) / values.reduce((a, b) => a + b, 0)).toFixed(1),
  };
}

test('--protocol team-270 prints their two windows and their results template', (t) => {
  // Frames of varying length, a second process's rows, and one 900 ms frame inside the walking window.
  const rows = [];
  let at = 0;
  let i = 0;
  let slow = false;
  while (at < 400000) {
    const pid = i % 7 === 6 ? 9999 : 4242;
    let ms = [100, 50, 150, 100, 200][i % 5];
    if (pid === 4242 && !slow && at >= 300000) {
      ms = 900;
      slow = true;
    }
    rows.push({ pid, at, ms });
    at += ms;
    i += 1;
  }
  const csv =
    'Application,ProcessID,TimeInMs,MsBetweenPresents,MsGPUBusy\r\n' +
    rows.map((r) => `${r.pid === 4242 ? 'victory.exe' : 'other.exe'},${r.pid},${r.at},${r.ms},50`).join('\r\n') + '\r\n';
  const frameAt = (ts, draws) => frame({ draws }).replace('2026-11-25T00:00:00.000000Z', ts);
  const log = protocolFixture(t, {
    rows: csv,
    logLines: [
      'harness env: protocol=team-270',
      'harness env: route=manual-1',
      'volume: pid 4242, device 1: 100% -> 0%',
      frameAt('2026-11-26T09:59:59.000000Z', 1111),
      'harness: settled at 75 s (waiting 40, 4.0 fps)',
      frameAt('2026-11-26T10:01:30.000000Z', 5000),
      frameAt('2026-11-26T10:03:20.000000Z', 7000),
      'harness: victory exit=0',
    ],
  });
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  const values = (a, b) => rows.filter((r) => r.pid === 4242 && r.at >= a * 1000 && r.at < b * 1000).map((r) => r.ms);
  const standing = windowNumbers(values(155, 225));
  const walking = windowNumbers(values(225, 345));
  const line = (label, w, draws) =>
    `- ${label}: ${w.frames} frames, ${w.fps} fps, median ${w.median} ms, p90 ${w.p90} ms, p99 ${w.p99} ms, 1% low ${w.low1} fps, worst ${w.worst} ms, ${draws} draws/frame, GPU busy ${w.gpu}%`;
  assert.match(result.stdout, /^## Protocol team-270 \(their 270 s test: stand to \+150, walk and turn to \+270, both after the settle\)$/m);
  // The draws anchor is the frame line before the settle marker: the standing window holds 5000, the walking 7000.
  assert.ok(result.stdout.includes(line('standing (+80..+150)', standing, '5000.0')), result.stdout);
  assert.ok(result.stdout.includes(line('walking (+150..+270)', walking, '7000.0')), result.stdout);
  assert.ok(result.stdout.includes("1% low: 1000 over the mean of the slowest 1% of frames; the windows' clock is within about 1 to 2 s of the settle"));
  // Their template: the route from the log, GPU busy as a percentage, no memory source in this log.
  const trow = (name, a, b) => `${name.padEnd(20)}${a.padEnd(24)}${b}`;
  assert.match(result.stdout, /^## Their results template$/m);
  assert.ok(result.stdout.includes('Standing (80-150 s)     Walking (150-270 s)'), result.stdout);
  assert.ok(result.stdout.includes(trow('FPS (average):', standing.fps, walking.fps)));
  assert.ok(result.stdout.includes(trow('Worst frame:', standing.worst, walking.worst)));
  assert.ok(result.stdout.includes(trow('GPU busy:', `${standing.gpu}%`, `${walking.gpu}%`)));
  assert.ok(result.stdout.includes('Scene:              route manual-1'), result.stdout);
  assert.ok(result.stdout.includes('Peak RAM (private): n/a  Peak VRAM: n/a  (this process, sampled about every 5 s, diagnostics builds only)'));
});

test('a run that ends before a window is spanned refuses that window with its reason', (t) => {
  const rows = [];
  let at = 0;
  while (at < 235000) {
    rows.push(`victory.exe,4242,${at},100,50`);
    at += 100;
  }
  const csv = 'Application,ProcessID,TimeInMs,MsBetweenPresents,MsGPUBusy\r\n' + rows.join('\r\n') + '\r\n';
  const log = protocolFixture(t, {
    rows: csv,
    logLines: [
      'harness env: protocol=team-270',
      'volume: pid 4242, device 1: 100% -> 0%',
      'harness: settled at 75 s (waiting 40, 4.0 fps)',
      'harness: victory exit=0',
    ],
  });
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /^- standing \(\+80\.\.\+150\): \d+ frames,/m);
  assert.match(result.stdout, /^- walking \(\+150\.\.\+270\): n\/a \(the run ended \d+ s short of it\)$/m);
});

test('rows of another process are not pooled, and the running clock is the game process own', (t) => {
  // No time column: the clock is the running sum of the game process's own frame times. Another process's interval
  // is its own, not the game's, so its rows must not advance the clock.
  const rows = [];
  for (let i = 0; i < 3200; i += 1) {
    const pid = i % 14 === 13 ? 9999 : 4242;
    rows.push({ pid, ms: pid === 9999 ? 1000 : 100 });
  }
  const csv =
    'Application,ProcessID,MsBetweenPresents,MsGPUBusy\r\n' +
    rows.map((r) => `${r.pid === 4242 ? 'victory.exe' : 'other.exe'},${r.pid},${r.ms},50`).join('\r\n') + '\r\n';
  let clock = 0;
  const game = [];
  for (const r of rows) {
    if (r.pid !== 4242) continue;
    clock += r.ms;
    game.push(clock);
  }
  const log = protocolFixture(t, {
    rows: csv,
    logLines: [
      'harness env: protocol=team-270',
      'volume: pid 4242, device 1: 100% -> 0%',
      'harness: settled at 75 s (waiting 40, 4.0 fps)',
      'harness: victory exit=0',
    ],
  });
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  const count = game.filter((t) => t >= 155000 && t < 225000).length;
  // With the other process's 1 s rows added to the clock the frames land elsewhere and this count changes.
  assert.ok(result.stdout.includes(`- standing (+80..+150): ${count} frames`), result.stdout);
});

// Peak RAM and VRAM come from the log's own lines: `memory: <n> MB private` for the process's private bytes, and
// the card's in-use figure in either build's wording.
test('the template takes peak RAM and peak VRAM from the log lines, in either wording', (t) => {
  const run = (lines) => {
    const log = fixture(t, ['harness env: protocol=team-270', 'harness: settled at 75 s (waiting 40, 4.0 fps)', ...lines, 'harness: victory exit=0']);
    const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
    assert.equal(result.status, 0, result.stderr);
    return result;
  };
  const note = '(this process, sampled about every 5 s, diagnostics builds only)';
  const ours = run([
    `${TS}memory: 8114 MB private (7900 MB of it in memory; the PC has 16708 MB free and 35963 MB left to commit)`,
    `${TS}the card: 1546 MB in use of a budget of 31419 MB, 0 MB of the PC's memory shared with it in use`,
    `${TS}memory: 9000 MB private (8800 MB of it in memory; the PC has 16000 MB free and 35000 MB left to commit)`,
    `${TS}the card: 2615 MB in use of a budget of 31419 MB`,
  ]);
  assert.ok(ours.stdout.includes(`Peak RAM (private): 9000 MB  Peak VRAM: 2615 MB  ${note}`), ours.stdout);
  const theirs = run([
    `${TS}memory: 10092 MB private (9800 MB of it in memory; the PC has 16708 MB free and 35963 MB left to commit)`,
    `${TS}video memory as the system counts it, with the memory line: the card's own: budget 31419 MB, in use 2908 MB, 15837 MB could still be reserved, 0 MB reserved`,
  ]);
  assert.ok(theirs.stdout.includes(`Peak RAM (private): 10092 MB  Peak VRAM: 2908 MB  ${note}`), theirs.stdout);
  const neither = run([`${TS}30.0 fps, 33.3 ms/frame (0.4 preparing: copies 0.2 ms 0.3 MB, textures 0.0 ms 0.0 MB; 0.0 on the GPU; 0.0 copying back 0.0 MB), 5000 draws/frame, dropped 0 without a host pipeline (0 being made) and 0 indirect`]);
  assert.ok(neither.stdout.includes(`Peak RAM (private): n/a  Peak VRAM: n/a  ${note}`), neither.stdout);
});

test('a capture that ends 2 s after the walking window is accepted', (t) => {
  const rows = [];
  let at = 0;
  while (at < 347000) {
    rows.push(`victory.exe,4242,${at},100,50`);
    at += 100;
  }
  const csv = 'Application,ProcessID,TimeInMs,MsBetweenPresents,MsGPUBusy\r\n' + rows.join('\r\n') + '\r\n';
  const log = protocolFixture(t, {
    rows: csv,
    logLines: [
      'harness env: protocol=team-270',
      'volume: pid 4242, device 1: 100% -> 0%',
      'harness: settled at 75 s (waiting 40, 4.0 fps)',
      'harness: victory exit=0',
    ],
  });
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  // The settle is 75 s, so the walking window ends at 345 s; the last row is at 346.9 s, inside the 1 s slack.
  assert.match(result.stdout, /^- walking \(\+150\.\.\+270\): \d+ frames,/m);
});

test('a start-up video-memory line alone is not a peak VRAM', (t) => {
  const log = fixture(t, [
    'harness env: protocol=team-270',
    'harness: settled at 75 s (waiting 40, 4.0 fps)',
    `${TS}video memory as the system counts it, right before the swap chain is made: the card's own: budget 31419 MB, in use 188 MB, 15837 MB could still be reserved, 0 MB reserved`,
    'harness: victory exit=0',
  ]);
  const result = spawnSync(process.execPath, [script, log], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 0, result.stderr);
  assert.ok(result.stdout.includes('Peak RAM (private): n/a  Peak VRAM: n/a  (this process, sampled about every 5 s, diagnostics builds only)'), result.stdout);
});
