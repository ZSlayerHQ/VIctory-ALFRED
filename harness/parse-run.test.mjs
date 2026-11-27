import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { isolatedEnv } from './test-isolation.mjs';
import { parseExtras, parseLogLines, parseTranslationFailures, stripControl } from './parse-run.mjs';
import { BUDGET_LINE, BUDGET_LINE_OTHERS, FRAME_LINE } from './fixtures/runtime-lines.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));

const HEAD =
  " WARN victory_gpu::pipeline: shader does not translate; its pipelines' draws are dropped";
const TS = '2026-11-24T20:27:48.000000Z';

const oldLine = (stage, hash, reason) =>
  `${TS}${HEAD} stage="${stage}" hash=${hash} error=converter shader.dxil v1: GFX10 code at 0x1234: ${reason}`;

const newLine = (stage, hash, reason, trailer = 'cached=false ms=9 user_ms=0 kernel_ms=0') =>
  `${TS}${HEAD} stage="${stage}" hash=${hash} error=GFX10 code at 0x288: ${reason} ${trailer}`;

test('old format: converter and GFX10 prefixes are stripped for every stage', () => {
  const ballot = 'a 64-lane wave ballot in divergent control flow is not supported';
  const reduction = 'a wave reduction in divergent control flow is not supported';
  const lds = 'LDS accesses without a uniform LDS fence in split wave64 is not supported';
  const crossLane = 'a cross-lane read in divergent control flow is not supported';

  const lines = [
    oldLine('compute', '0123456789abcdef', ballot),
    oldLine('mesh', 'bbbbbbbbbbbbbbbb', reduction),
    oldLine('pixel', 'cccccccccccccccc', lds),
    oldLine('vertex', 'dddddddddddddddd', crossLane),
  ];

  assert.deepEqual(parseTranslationFailures(lines), [
    { stage: 'compute', hash: '0123456789abcdef', reason: ballot },
    { stage: 'mesh', hash: 'bbbbbbbbbbbbbbbb', reason: reduction },
    { stage: 'pixel', hash: 'cccccccccccccccc', reason: lds },
    { stage: 'vertex', hash: 'dddddddddddddddd', reason: crossLane },
  ]);
});

test('new format: GFX10 prefix and cached/ms/user_ms/kernel_ms fields are stripped', () => {
  const reason = 'a 64-lane wave ballot in divergent control flow is not supported';
  const line = newLine('compute', '0123456789abcdef', reason);

  assert.deepEqual(parseTranslationFailures([line]), [
    { stage: 'compute', hash: '0123456789abcdef', reason },
  ]);
});

test('new format: any trailing run of word=value fields is stripped', () => {
  const reason = 'LDS accesses without a uniform LDS fence in split wave64 is not supported';
  const lines = [
    newLine('mesh', 'bbbbbbbbbbbbbbbb', reason, 'cached=true attempts=3 elapsed_ms=41'),
    newLine('pixel', 'cccccccccccccccc', reason, 'cached=false retries=0 note=a=b'),
  ];

  assert.deepEqual(parseTranslationFailures(lines), [
    { stage: 'mesh', hash: 'bbbbbbbbbbbbbbbb', reason },
    { stage: 'pixel', hash: 'cccccccccccccccc', reason },
  ]);
});

test('missing GFX10 prefix and missing converter prefix are both optional', () => {
  const ballot = 'a 64-lane wave ballot in divergent control flow is not supported';
  const reduction = 'a wave reduction in divergent control flow is not supported';

  const lines = [
    `${TS}${HEAD} stage="compute" hash=0123456789abcdef error=${ballot} cached=false ms=9`,
    `${TS}${HEAD} stage="pixel" hash=bbbbbbbbbbbbbbbb error=converter shader.dxil v1: ${reduction}`,
  ];

  assert.deepEqual(parseTranslationFailures(lines), [
    { stage: 'compute', hash: '0123456789abcdef', reason: ballot },
    { stage: 'pixel', hash: 'bbbbbbbbbbbbbbbb', reason: reduction },
  ]);
});

test('an = inside a reason is kept unless it is part of a trailing run', () => {
  const reason = 'operand x=y in divergent control flow is not supported';
  const line = `${TS}${HEAD} stage="compute" hash=0123456789abcdef error=GFX10 code at 0x10: ${reason} cached=false ms=1`;

  assert.deepEqual(parseTranslationFailures([line]), [
    { stage: 'compute', hash: '0123456789abcdef', reason },
  ]);
});

test('ANSI-coloured text input is still parsed', () => {
  const reason = 'a cross-lane read in divergent control flow is not supported';
  const line =
    '\x1b[2m' + newLine('compute', '0123456789abcdef', reason) + '\x1b[0m';

  assert.deepEqual(parseTranslationFailures(line), [
    { stage: 'compute', hash: '0123456789abcdef', reason },
  ]);
});

test('the shared stripper removes any control sequence, not only colour codes', () => {
  const codes = [
    '\x1b[0m', // colour reset
    '\x1b[38;5;196m', // 256-colour
    '\x1b[?25l', // cursor hide (private parameter)
    '\x1b]0;a window title\x07', // OSC, BEL-terminated
    '\x1b]2;a window title\x1b\\', // OSC, ST-terminated
    '\x1b(B', // ESC ( B, the default character set
    '\x1b[1;2H', // cursor position
    '\x9b31m', // 8-bit CSI
  ];
  assert.equal(stripControl(`hello${codes.join('')} world`), 'hello world');
  assert.equal(stripControl('no codes here'), 'no codes here');
});

test('non-failure lines are ignored and failure order is kept', () => {
  const first = 'a 64-lane wave ballot in divergent control flow is not supported';
  const second = 'a wave reduction in divergent control flow is not supported';
  const lines = [
    `${TS}  INFO victory_gpu::pipeline: shader translated stage="compute" hash=aaaaaaaaaaaaaaaa user_data=0 samplers=0`,
    oldLine('compute', '0123456789abcdef', first),
    newLine('pixel', 'bbbbbbbbbbbbbbbb', second),
    `${TS}  INFO victory_gpu::host: 60.0 fps, 16.6 ms/frame`,
  ];

  assert.deepEqual(parseTranslationFailures(lines), [
    { stage: 'compute', hash: '0123456789abcdef', reason: first },
    { stage: 'pixel', hash: 'bbbbbbbbbbbbbbbb', reason: second },
  ]);
});

const SYNTH_TS = '2026-11-20T21:08:23.0000000Z';
const driverLine = (id, text) => `driver event: ${SYNTH_TS} id=${id} ${text}`;

test('driver events: a count per kind and distinct ESR value sets without register addresses', () => {
  const lines = [
    driverLine(13, 'Graphics SM Warp Exception on (GPC 1, TPC 4, SM 0): MMU Fault'),
    driverLine(13, 'Graphics SM Global Exception on (GPC 10, TPC 7, SM 0): Multiple Warp Errors'),
    driverLine(
      13,
      'Graphics Exception: ESR 0x555730=0x82a0017 0x555734=0x0 0x555728=0x1c813b60 0x55572c=0x1104',
    ),
    driverLine(
      13,
      'Graphics Exception: ESR 0x5ac730=0x82a0017 0x5ac734=0x0 0x5ac728=0x1c813b60 0x5ac72c=0x1104',
    ),
    driverLine(
      13,
      'Graphics Exception: ESR 0x5077b0=0x82d0017 0x5077b4=0x4 0x5077a8=0x1c813b60 0x5077ac=0x1104',
    ),
    driverLine(153, 'Error occurred on GPUID:'),
  ];

  const { summary } = parseLogLines(lines, 'synthetic.log');
  assert.deepEqual(summary.driverEvents, {
    total: 6,
    counts: { mmuFault: 1, multipleWarpErrors: 1, otherId13: 3, id153: 1 },
    // Registers differ per shader core; only the values are kept, and equal sets merge.
    esrValueSets: [
      { values: ['0x82a0017', '0x0', '0x1c813b60', '0x1104'], count: 2 },
      { values: ['0x82d0017', '0x4', '0x1c813b60', '0x1104'], count: 1 },
    ],
  });
});

test('settings come only from harness env header lines, never from matching body text', () => {
  const body = [
    `${TS}  WARN victory_gpu::shader_cache: VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner limit=999`,
    `${TS}  INFO victory_gpu::host: Wave64 compute lowering mode selected wave64_mode=Native`,
    `${TS}  INFO victory_gpu::host: VIctory settings: skipped pipelines=feedfacefeedface gpu timing=on copy mode=none aftermath=off`,
    `${TS} ERROR victory_gpu::host: host GPU device removed; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command, and VICTORY_D3D12_GBV=1 validates it`,
  ];
  const header = [
    'harness build: abc profile=diag console=series-x',
    'harness env: VICTORY_SHADER_LOOP_LIMIT=32',
    'harness env: VICTORY_WAVE64_EMULATE=emulate',
    'harness env: VICTORY_SKIP_PIPELINES=0123456789abcdef',
    'harness env: VICTORY_GPU_TIMING=1',
    'harness env: VICTORY_COPY_MODE=plain',
    'harness env: VICTORY_AFTERMATH=E:/captures/synthetic',
    'harness env: record=on',
  ];

  assert.deepEqual(parseExtras([...header, ...body], null, null).settings, {
    skippedPipelines: '0123456789abcdef',
    loopLimit: 32,
    wave64: 'emulate',
    gpuTiming: true,
    d3d12Debug: false,
    d3d12Gbv: false,
    copyMode: 'plain',
    aftermath: true,
    record: true,
  });

  // A header with none of the flags records that they were off.
  const bareHeader = [
    'harness build: abc profile=diag console=series-x',
    'harness env: VICTORY_LOG=info',
  ];
  const bareSettings = parseExtras([...bareHeader, ...body], null, null).settings;
  assert.equal(bareSettings.d3d12Debug, false);
  assert.equal(bareSettings.d3d12Gbv, false);
  assert.equal(bareSettings.gpuTiming, false);
  assert.equal(bareSettings.aftermath, false);
  assert.equal(bareSettings.record, false);

  // No header at all: nothing is known, and body text is never a substitute.
  assert.deepEqual(parseExtras(body, null, null).settings, {
    skippedPipelines: null,
    loopLimit: null,
    wave64: null,
    gpuTiming: null,
    d3d12Debug: null,
    d3d12Gbv: null,
    copyMode: null,
    aftermath: null,
    record: null,
  });
});

test('summarize-run: the Driver events section reports the counts and the ESR values', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'driver-events-summary-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const log = path.join(dir, '20261124-210000-events.log');
  fs.writeFileSync(
    log,
    [
      driverLine(13, 'Graphics SM Warp Exception on (GPC 1, TPC 4, SM 0): MMU Fault'),
      driverLine(13, 'Graphics SM Global Exception on (GPC 10, TPC 7, SM 0): Multiple Warp Errors'),
      driverLine(
        13,
        'Graphics Exception: ESR 0x555730=0x82a0017 0x555734=0x0 0x555728=0x1c813b60 0x55572c=0x1104',
      ),
      driverLine(153, 'Error occurred on GPUID:'),
      '',
    ].join('\n'),
  );

  const result = spawnSync(process.execPath, [path.join(here, 'summarize-run.mjs'), log], {
    encoding: 'utf8',
    env: isolatedEnv(),
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /## Driver events/);
  assert.match(result.stdout, /- MMU Fault: 1/);
  assert.match(result.stdout, /- Multiple Warp Errors: 1/);
  assert.match(result.stdout, /- other id 13 texts: 1/);
  assert.match(result.stdout, /- id 153: 1/);
  assert.match(result.stdout, /- ESR values: 0x82a0017 0x0 0x1c813b60 0x1104 x1/);

  const json = JSON.parse(
    spawnSync(process.execPath, [path.join(here, 'summarize-run.mjs'), log, '--json'], {
      encoding: 'utf8',
      env: isolatedEnv(),
    }).stdout,
  );
  assert.equal(json.driverEvents.total, 4);
  assert.deepEqual(json.driverEvents.counts, {
    mmuFault: 1,
    multipleWarpErrors: 1,
    otherId13: 1,
    id153: 1,
  });
});

test('a title unhandled exception is a failure and ends the run as a title crash', () => {
  const lines = [
    '2026-11-24T12:00:00.000000Z  INFO victory_gpu::host: 30.0 fps, 33.3 ms/frame (0.3 preparing: copies 0.2 ms 0.4 MB, textures 0.0 ms 0.0 MB; 0.7 on the GPU; 0.0 copying back 0.0 MB), 12 draws/frame, dropped 0 without a host pipeline (0 being made) and 1 indirect',
    'VIctory: unhandled exception 0xc0000005 at victory-image-0123456789abcdef.dll+0x1 (write 0x0: nothing mapped there)',
  ];
  const { summary, firstTs, lastTs } = parseLogLines(lines, 'synthetic.log');
  assert.equal(summary.failures.length, 1);
  assert.match(summary.failures[0], /unhandled exception/);

  const extras = parseExtras(lines, firstTs, lastTs);
  assert.equal(extras.ending.kind, 'title_crashed');
  assert.equal(extras.ending.text, 'title crashed');
});

test('a stopped command with no marker reports an unknown pipeline, not a later dispatch hash', () => {
  const line =
    '2026-11-24T12:00:10.000000Z ERROR victory_gpu::host: host GPU device removed after a submission of 2 lists; ' +
    'breadcrumbs (done/count, list) ["1/1 None"]; a list stopped at command 260 of 560 (op 6), commands 259..261 between its barriers:; ' +
    'command 260 (op 6): no marker; a list stopped at command 0 of 3 (op 47); list 3 of 12, of its 4: ' +
    'dispatch 33x33x6 [compute fedcba9876543210] | fedcba9876543210: u4=0x1 [heap 1: null]';
  const { details, firstTs, lastTs } = parseLogLines([line], 'synthetic.log');
  assert.equal(details.hangs.length, 1);
  assert.equal(details.hangs[0].command, 'no marker');
  assert.equal(details.hangs[0].pipelineUnknown, 'no marker');

  const extras = parseExtras([line], firstTs, lastTs);
  assert.equal(extras.hung.pipelineHash, null);
  assert.equal(extras.hung.stage, null);
  assert.equal(extras.hung.pipelineUnknown, 'no marker');
  assert.deepEqual(extras.hung.nullSlots, []);
});

test('summarize-run: a markerless stopped command names no pipeline', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'no-marker-summary-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const log = path.join(dir, '20261124-210000-nomarker.log');
  fs.writeFileSync(
    log,
    [
      '2026-11-24T12:00:10.000000Z ERROR victory_gpu::host: host GPU device removed after a submission of 2 lists; ' +
        'breadcrumbs (done/count, list) ["1/1 None"]; a list stopped at command 260 of 560 (op 6), commands 259..261 between its barriers:; ' +
        'command 260 (op 6): no marker; a list stopped at command 0 of 3 (op 47); list 3 of 12, of its 4: ' +
        'dispatch 33x33x6 [compute fedcba9876543210] | fedcba9876543210: u4=0x1 [heap 1: null]',
      '',
    ].join('\n'),
  );

  const result = spawnSync(process.execPath, [path.join(here, 'summarize-run.mjs'), log], {
    encoding: 'utf8',
    env: isolatedEnv(),
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /stopped at command 260\/560 \(op 6\): pipeline unknown \(no marker\)/);
  assert.equal(result.stdout.includes('fedcba9876543210'), false, result.stdout);
});

test('a harness-ended run says so, ahead of a device removal or a crash in the same log', () => {
  const lines = [
    '2026-11-24T12:00:00.000000Z  INFO victory_gpu::host: 30.0 fps, 33.3 ms/frame (0.3 preparing: copies 0.2 ms 0.4 MB, textures 0.0 ms 0.0 MB; 0.7 on the GPU; 0.0 copying back 0.0 MB), 12 draws/frame, dropped 0 without a host pipeline (0 being made) and 1 indirect',
    '2026-11-24T12:00:09.000000Z ERROR victory_gpu::host: host GPU device removed after a submission of 2 lists',
    'VIctory: unhandled exception 0xc0000005 at victory-image-0123456789abcdef.dll+0x1',
    'harness: ended by --heavy-seconds after 30 s of heavy scene',
  ];

  const { summary, firstTs, lastTs } = parseLogLines(lines, 'synthetic.log');
  assert.equal(summary.durationSeconds, 9);
  const extras = parseExtras(lines, firstTs, lastTs);
  assert.deepEqual(extras.ending, {
    kind: 'harness_ended',
    seconds: 30,
    frame: null,
    text: 'ended by the harness after 30 s of heavy scene',
  });

  // Without the line the same log is a device removal, so the new line is what changes the ending.
  assert.equal(parseExtras(lines.slice(0, 3), firstTs, lastTs).ending.kind, 'device_removed');
});

test('a settled run carries its settle time and ends on the settled clock', () => {
  const lines = [
    '2026-11-24T12:00:00.000000Z  INFO victory_gpu::host: 30.0 fps, 33.3 ms/frame (0.3 preparing: copies 0.2 ms 0.4 MB, textures 0.0 ms 0.0 MB; 0.7 on the GPU; 0.0 copying back 0.0 MB), 12 draws/frame, dropped 0 without a host pipeline (0 being made) and 1 indirect',
    'harness: settled at 57 s (waiting 1558, 2.9 fps)',
    'harness: ended by --settled-seconds after 75 s of settled scene',
  ];
  const { firstTs, lastTs } = parseLogLines(lines, 'synthetic.log');
  const extras = parseExtras(lines, firstTs, lastTs);
  assert.deepEqual(extras.settled, { seconds: 57, waiting: 1558, fps: 2.9 });
  assert.equal(extras.settledTimeout, false);
  assert.equal(extras.ending.text, 'ended by the harness after 75 s of settled scene');

  const timedOut = [
    'harness: ended by --settled-timeout after 600 s: the scene never settled',
  ];
  const timeout = parseExtras(timedOut, firstTs, lastTs);
  assert.equal(timeout.settled, null);
  assert.equal(timeout.settledTimeout, true);
});

test('host mesh pipeline refused counts as a refused pipeline too', () => {
  const lines = [
    `${TS}  WARN victory_gpu::pipeline::host: host pipeline refused error=The parameter is incorrect. (0x80070057) shaders=[compute aaaaaaaaaaaaaaaa]`,
    `${TS}  WARN victory_gpu::pipeline::host: host mesh pipeline refused error=The parameter is incorrect. (0x80070057) shaders=[mesh 1111222233334444, pixel 5555666677778888]`,
    `${TS}  WARN victory_gpu::pipeline::host: host root signature refused error=The parameter is incorrect. (0x80070057) shaders=[pixel bbbbbbbbbbbbbbbb]`,
  ];
  const { summary } = parseLogLines(lines, 'synthetic.log');
  assert.equal(summary.refusedPipelines.count, 2);
  assert.deepEqual(summary.refusedPipelines.hashes, [
    'compute aaaaaaaaaaaaaaaa',
    'mesh 1111222233334444, pixel 5555666677778888',
  ]);
});

test('a title-chosen exit is its own ending and a failure when the code is not 0', () => {
  const lines = [
    `${TS}  WARN victory_hle::modules::kernel: title ended the process: call ExitProcess, code 127, from game.exe+0x1234`,
    'harness: victory exit=127',
  ];
  const { summary, firstTs, lastTs } = parseLogLines(lines, 'synthetic.log');
  assert.deepEqual(parseExtras(lines, firstTs, lastTs).ending, {
    kind: 'title_ended',
    seconds: null,
    frame: null,
    text: 'the title ended the process: ExitProcess, code 127',
  });
  assert.equal(summary.failures.length, 1);
  assert.match(summary.failures[0], /title ended the process: call ExitProcess, code 127/);
});

test('a title exit with code 0 is an ending of its own, not a failure', () => {
  const lines = [
    `${TS}  WARN victory_hle::modules::kernel: title ended the process: call exit, code 0, from game.exe+0x10`,
    'harness: victory exit=0',
  ];
  const { summary, firstTs, lastTs } = parseLogLines(lines, 'synthetic.log');
  assert.equal(summary.failures.length, 0);
  assert.equal(
    parseExtras(lines, firstTs, lastTs).ending.text,
    'the title ended the process: exit, code 0',
  );
});

test('runtime exit: code 0 is a clean exit beside the older exit code 0 form', () => {
  assert.equal(
    parseExtras([`${TS}  INFO victory: runtime exit: code 0`], null, null).ending.kind,
    'clean_exit',
  );
  assert.equal(
    parseExtras([`${TS}  INFO victory: exit code 0`], null, null).ending.kind,
    'clean_exit',
  );
});

test('a non-zero harness exit with no failure line is flagged, not "log ends"', () => {
  assert.deepEqual(parseExtras(['harness: victory exit=127'], null, null).ending, {
    kind: 'exited_without_a_failure_line',
    seconds: null,
    frame: null,
    text: 'exited with code 127 and no failure line: check what else was running',
  });
});

test('a log with no closing harness line is cut short, not "no failure line"', () => {
  assert.deepEqual(parseExtras(['2026-11-24T12:00:00.000000Z  INFO game: a frame'], null, null).ending, {
    kind: 'log_cut_short',
    seconds: null,
    frame: null,
    text: 'log cut short: no closing harness line',
  });
  // The harness's own end markers count as a closing line.
  for (const line of [
    'harness: victory exit=1',
    'harness: ended by --heavy-seconds after 30 s of heavy scene',
    'harness: ended by --settled-timeout after 600 s: the scene never settled',
    'harness: ended: disk below 5 GB',
  ]) {
    assert.notEqual(parseExtras([line], null, null).ending.kind, 'log_cut_short', line);
  }
});

test('the free-space guard end line is its own ending', () => {
  assert.deepEqual(parseExtras(['harness: ended: disk below 5 GB'], null, null).ending, {
    kind: 'disk_low',
    seconds: null,
    frame: null,
    text: 'ended: disk below 5 GB',
  });
});

test('the watchdog stall line is its own ending, ahead of the harness exit', () => {
  const lines = [
    'harness: watchdog: the frame counter stopped at frame 4321 for 47 s; ending the run',
    'harness: victory exit=1',
  ];
  assert.deepEqual(parseExtras(lines, null, null).ending, {
    kind: 'watchdog',
    seconds: 47,
    frame: 4321,
    text: 'ended: watchdog (frame 4321, 47 s)',
  });
  // A reader must not take this for a normal end, even when a hold timer's line follows it.
  assert.equal(
    parseExtras(
      [...lines, 'harness: ended by --heavy-seconds after 30 s of heavy scene'],
      null,
      null,
    ).ending.kind,
    'watchdog',
  );
});

test('warn-mode stalls are counted, with the longest including the resume length', () => {
  const lines = [
    'harness: watchdog: the frame counter stopped at frame 4321 for 47 s; NOT ending the run (warn mode)',
    'harness: watchdog: frames resumed at frame 4328 after 52 s',
    'harness: watchdog: the frame counter stopped at frame 4400 for 45 s; NOT ending the run (warn mode)',
    'harness: victory exit=0',
  ];
  const extras = parseExtras(lines, null, null);
  assert.deepEqual(extras.watchdog, { stalls: 2, longestSeconds: 52 });
  assert.equal(extras.ending.kind, 'clean_exit', 'a warn line is not an ending');
  // A run with no stall line carries no watchdog field at all.
  assert.equal(parseExtras(['harness: victory exit=0'], null, null).watchdog, null);
});

test('the harness exit does not overrule a failure line or a clean exit', () => {  // A crash with the same code keeps the crash's ending.
  assert.equal(
    parseExtras(
      [
        'VIctory: unhandled exception 0xc0000005 at victory-image-0123456789abcdef.dll+0x1',
        'harness: victory exit=127',
      ],
      null,
      null,
    ).ending.kind,
    'title_crashed',
  );
  // A deliberate end by --heavy-seconds still wins over its non-zero code.
  assert.equal(
    parseExtras(
      ['harness: ended by --heavy-seconds after 30 s of heavy scene', 'harness: victory exit=1'],
      null,
      null,
    ).ending.kind,
    'harness_ended',
  );
  assert.equal(parseExtras(['harness: victory exit=0'], null, null).ending.kind, 'clean_exit');
});

test('summarize-run: the screen record line names the video, its frames and its sheets', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'screen-record-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const capture = path.join(dir, 'capture');
  fs.mkdirSync(path.join(capture, 'sheets'), { recursive: true });
  const made = spawnSync(
    'ffmpeg',
    ['-hide_banner', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=size=320x180:rate=2:duration=3', '-pix_fmt', 'yuv420p', path.join(capture, 'screen.mp4')],
    { encoding: 'utf8' },
  );
  assert.equal(made.status, 0, made.stderr);
  fs.writeFileSync(path.join(capture, 'sheets', 'sheet-001.jpg'), Buffer.from([0xff, 0xd8, 0xff]));

  const log = path.join(dir, '20261124-230000-record.log');
  fs.writeFileSync(
    log,
    [
      'harness build: abc profile=diag console=series-x',
      'harness env: record=on',
      `capture: ${capture}`,
      'harness: victory exit=0',
      '',
    ].join('\n'),
  );
  const result = spawnSync(process.execPath, [path.join(here, 'summarize-run.mjs'), log], {
    encoding: 'utf8',
    env: isolatedEnv(),
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Screen record: .*screen\.mp4 \(6 frames, 1 sheets\)/);

  // No video in the capture folder: none.
  const other = path.join(dir, '20261124-230001-norecord.log');
  fs.writeFileSync(
    other,
    [
      'harness build: abc profile=diag console=series-x',
      'harness env: record=off',
      `capture: ${path.join(dir, 'empty')}`,
      'harness: victory exit=0',
      '',
    ].join('\n'),
  );
  const none = spawnSync(process.execPath, [path.join(here, 'summarize-run.mjs'), other], {
    encoding: 'utf8',
  });
  assert.equal(none.status, 0, none.stderr);
  assert.match(none.stdout, /Screen record: none/);
});

const quietStart = 'harness: load at start: cpu 4% over 5 s, free memory 42.1 GB of 63.8, commit 38.2 of 64.0 GB, builders 0';
const quietEnd = 'harness: load at end: cpu 7% over 5 s, free memory 40.0 GB of 63.8, commit 39.0 of 64.0 GB, builders 0';

test('the load lines give the summary a comparable verdict', () => {
  assert.equal(parseLogLines([quietStart, quietEnd], 'run.log').summary.comparable, 'yes');
  assert.equal(
    parseLogLines(['harness: load at start: cpu 41% over 5 s, free memory 42.1 GB of 63.8, commit 38.2 of 64.0 GB, builders 0', quietEnd], 'run.log')
      .summary.comparable,
    'no (cpu 41% is over 25% at start)',
  );
  assert.equal(
    parseLogLines([quietStart, 'harness: load at end: cpu 7% over 5 s, free memory 9.2 GB of 63.8, commit 39.0 of 64.0 GB, builders 2'], 'run.log')
      .summary.comparable,
    'no (free memory 9.2 GB is under 12 GB at end; 2 builders is over 0 at end)',
  );
  assert.equal(parseLogLines([], 'run.log').summary.comparable, 'unknown (no load lines)');
});

test('the comparable thresholds are variables with the documented defaults', () => {
  const hot = 'harness: load at start: cpu 41% over 5 s, free memory 42.1 GB of 63.8, commit 38.2 of 64.0 GB, builders 0';
  const parsed = parseLogLines([hot, quietEnd], 'run.log', {
    env: { HARNESS_MAX_CPU: '50', HARNESS_MIN_FREE_MEM_GB: '12', HARNESS_MAX_BUILDERS: '0' },
  });
  assert.equal(parsed.summary.comparable, 'yes');
  assert.deepEqual(parsed.summary.load.limits, { maxCpu: 50, minFreeMemGb: 12, maxBuilders: 0 });
  assert.equal(parsed.summary.load.start.cpu, 41);
});

const OLD_FRAME =
  '2026-11-24T12:00:00.000000Z  INFO victory_gpu::host: 30.0 fps, 33.3 ms/frame ' +
  '(0.3 preparing: copies 0.2 ms 0.4 MB, textures 0.0 ms 0.0 MB; 0.7 on the GPU; ' +
  '0.0 copying back 0.0 MB), 12 draws/frame, dropped 0 without a host pipeline (0 being made) and 1 indirect';
// The text the runtime's frame-line format writes from job/honest-counters on, colour codes included.
const NEW_FRAME_COLOURED =
  '\x1b[2m2026-11-26T05:02:11.250000Z\x1b[0m \x1b[32m INFO\x1b[0m ' +
  '\x1b[2mvictory_gpu::host\x1b[0m\x1b[2m:\x1b[0m ' +
  'frame 18432, 27.4 fps, 36.5 ms/frame (0.4 preparing: copies 0.3 ms 1.2 MB, ' +
  'textures hashed 0.2 ms 0.1 MB; 1.7 on the GPU; 0.1 copying back 0.9 MB merged into ' +
  "title memory (host time, fetches included), 0.0 MB fetched from copies in the card's memory), " +
  '5000 draws/frame, dropped 3 without a host pipeline (1 being made) and 7 indirect';
const presentLine = (frame) =>
  `2026-11-24T12:00:00.500000Z  INFO victory_gpu::queue: frame presented (since the last line: draws and ` +
  `dispatches, and those run on the host) frame=${frame} draws=12 drawn=12 dispatches=0 dispatched=0`;

test('frame line: the old form parses, and its last presented frame is the present line fallback', () => {
  const { summary } = parseLogLines([OLD_FRAME, presentLine(8192)], 'old.log');
  const line = summary.lastFrameLine;
  assert.equal(line.frame, null);
  assert.equal(line.fps, 30);
  assert.equal(line.ms, 33.3);
  assert.equal(line.draws, 12);
  assert.equal(line.dropped, 0);
  assert.equal(line.making, 0);
  assert.equal(line.indirect, 1);
  assert.equal(line.texturesMs, 0);
  assert.equal(line.texturesMb, 0);
  assert.equal(summary.lastFrame.frame, 8192);
  assert.equal(summary.lastFrame.draws, 12);
});

test('frame line: the new form parses its frame number and "textures hashed" figures', () => {
  const { summary } = parseLogLines([stripControl(NEW_FRAME_COLOURED)], 'new.log');
  const line = summary.lastFrameLine;
  assert.equal(line.frame, 18432);
  assert.equal(line.fps, 27.4);
  assert.equal(line.ms, 36.5);
  assert.equal(line.draws, 5000);
  assert.equal(line.dropped, 3);
  assert.equal(line.making, 1);
  assert.equal(line.indirect, 7);
  assert.equal(line.preparing, 0.4);
  assert.equal(line.copiesMs, 0.3);
  assert.equal(line.copiesMb, 1.2);
  assert.equal(line.texturesMs, 0.2);
  assert.equal(line.texturesMb, 0.1);
  assert.equal(line.hostWaitsMs, 1.7);
  assert.equal(line.backMs, 0.1);
  assert.equal(line.backMb, 0.9);
  // No "frame presented" line is needed: the frame line carries the count itself.
  assert.equal(summary.lastFrame.frame, 18432);
  assert.equal(summary.lastFrame.draws, null);
});

test('a log mixing both frame-line forms parses, and the newest numbered line wins', () => {
  const mixed = parseLogLines(
    [OLD_FRAME, presentLine(8192), stripControl(NEW_FRAME_COLOURED), OLD_FRAME],
    'mixed.log',
  ).summary;
  assert.equal(mixed.fps.samples, 3);
  assert.equal(mixed.lastFrameLine.frame, null, 'the last line is the old form');
  assert.equal(mixed.lastFrame.frame, 18432, 'the newest numbered line, not the old present line');
});

// The copystat lines exactly as copystat.rs formats them, old and new.
const OLD_COPIES =
  '2026-11-20T04:17:12.000000Z  INFO victory_gpu::copystat: copies per frame: 1.0 submissions; ' +
  '2 title copies 3.0 MB; table fills 4.0 MB asked, 5.0 MB filled in 0.10 ms; 6 refreshes 7.0 MB, ' +
  '8 skipped 9.0 MB; 10 runs 0.20 ms; 11 queries 0.30 ms, moves 0.40 ms, 12.0 MB unmapped; ' +
  '13 page faults; 14 watches 0.50 ms (1 failed 0.01 ms), 15.0 MB checked, 16.0 MB unchanged; ' +
  '18 fallbacks; 19 guard faults; 20 idle copies kept, 21 let go by pressure; textures 22.0 MB watched ' +
  'and changed, 23.0 MB hashed, 24.0 MB of them changed';
const NEW_COPIES =
  '2026-11-26T05:02:11.800000Z  INFO victory_gpu::copystat: copies per frame: 1.0 submissions; ' +
  '2 title copies 3.0 MB; table fills 4.0 MB asked, 5.0 MB filled in 0.10 ms; ' +
  'descriptor-copy sync 0.25 ms (12 records); 6 refreshes 7.0 MB, 8 skipped 9.0 MB; ' +
  '10 runs 0.20 ms; 11 queries 0.30 ms, moves 0.40 ms, 12.0 MB unmapped; 13 page faults; ' +
  '14 watches 0.50 ms (1 failed 0.01 ms), 15.0 MB asked about (repeats counted), 16.0 MB unchanged, ' +
  '17 distinct pages since the last present; 18 fallbacks; 19 guard faults; 20 idle copies kept, ' +
  '21 let go by pressure; textures 22.0 MB watched and changed, 23.0 MB hashed, 24.0 MB of them changed';
const OLD_RECORDING =
  '2026-11-20T04:17:12.000000Z  INFO victory_gpu::copystat: recording per frame (105 frames): ' +
  '31 list calls 5.2 ms (all recording threads); 0.0 ms waiting for pipelines; dynamic slots: ' +
  '4 tables asked, 0 refused, 0.0 ms searching; notes 0.0 ms; shader-written ranges: 0 steps in ' +
  '0 submissions, snapshots 0.0 ms, 0.0 MB compared back; sparse reads 0.0 ms; guards 0.7 ms; ' +
  '0.0 MB copied whole (memory not watched); 0.0 frame waits sleeping 0.0 ms (all threads); ' +
  'ranges of shaders that cannot store to buffers: 0 steps 0.0 MB left out';
const NEW_RECORDING =
  '2026-11-26T05:02:11.800000Z  INFO victory_gpu::copystat: recording per frame (105 frames): ' +
  '31 list calls 5.2 ms (all recording threads); 0.0 ms waiting for pipelines; dynamic slots: ' +
  '4 tables asked, 0 refused, 0.0 ms searching; notes 0.0 ms; shader-written ranges: 0 steps in ' +
  '0 submissions, snapshots 0.0 ms, 0.0 MB compared back; sparse reads 0.0 ms; ' +
  "42 guards in 0.7 ms (the fault handler's included); 0.0 MB copied whole (memory not watched); " +
  '0.0 frame waits sleeping 0.0 ms (all threads); ranges of shaders that cannot store to buffers: ' +
  '0 steps 0.0 MB left out';

test('copy stats: the old "MB checked" and "guards X ms" forms parse', () => {
  const { summary } = parseLogLines([OLD_COPIES, OLD_RECORDING], 'old-copystat.log');
  assert.deepEqual(summary.copyStats, {
    descriptorCopySyncMs: null,
    descriptorCopyRecords: null,
    distinctPages: null,
    guardCount: null,
    guardsMs: 0.7,
    askedAboutMb: 15,
  });
});

test('copy stats: the new forms parse the distinct pages, guard count and descriptor-copy sync', () => {
  const { summary } = parseLogLines([NEW_COPIES, NEW_RECORDING], 'new-copystat.log');
  assert.deepEqual(summary.copyStats, {
    descriptorCopySyncMs: 0.25,
    descriptorCopyRecords: 12,
    distinctPages: 17,
    guardCount: 42,
    guardsMs: 0.7,
    askedAboutMb: 15,
  });
});

test('copy stats: a log mixing the two copystat forms fills what each line carries', () => {
  const oldThenNew = parseLogLines([OLD_COPIES, NEW_RECORDING], 'mixed-copystat.log').summary;
  assert.equal(oldThenNew.copyStats.descriptorCopySyncMs, null);
  assert.equal(oldThenNew.copyStats.distinctPages, null);
  assert.equal(oldThenNew.copyStats.guardCount, 42, 'the new recording line');
  assert.equal(oldThenNew.copyStats.askedAboutMb, 15, 'the old "MB checked"');
  const newThenOld = parseLogLines([NEW_COPIES, OLD_RECORDING], 'mixed-copystat.log').summary;
  assert.equal(newThenOld.copyStats.descriptorCopySyncMs, 0.25);
  assert.equal(newThenOld.copyStats.distinctPages, 17);
  assert.equal(newThenOld.copyStats.guardCount, null);
  assert.equal(newThenOld.copyStats.guardsMs, 0.7, 'the old "guards X ms"');
});

test('observers and the switches set are read from their start-up lines', () => {
  const lines = [
    '2026-11-26T05:02:10.000000Z  INFO victory_gpu::host: observers on: the debug layer, ' +
      'GPU-based validation, DRED, GPU timing, the memory map walk, the counting allocator',
    '2026-11-26T05:02:10.100000Z  INFO victory_gpu::switches: diagnostic switch set ' +
      'switch=VICTORY_D3D12_DEBUG value=1 on=true',
    '2026-11-26T05:02:10.200000Z  INFO victory_gpu::switches: diagnostic switch set ' +
      'switch=VICTORY_MEMORY_MAP value=0 on=false',
  ];
  const { summary } = parseLogLines(lines, 'observers.log');
  assert.deepEqual(summary.observers, [
    'the debug layer',
    'GPU-based validation',
    'DRED',
    'GPU timing',
    'the memory map walk',
    'the counting allocator',
  ]);
  assert.deepEqual(summary.switches, [
    { name: 'VICTORY_D3D12_DEBUG', value: '1', on: true },
    { name: 'VICTORY_MEMORY_MAP', value: '0', on: false },
  ]);
});

test('a log without the observers or switch lines reports none of them', () => {
  const { summary } = parseLogLines([OLD_FRAME], 'plain.log');
  assert.equal(summary.observers, null);
  assert.deepEqual(summary.switches, []);
  assert.equal(summary.copyStats, null);
});

test('the speed work\'s "buffer copies per frame" line is not taken for a copystat line', () => {
  const bufferCopies =
    '2026-11-25T00:00:00.000000Z  INFO victory_gpu::host: buffer copies per frame: 5 on the card ' +
    'from host copies (2.0 MB), 1 read from title memory on the CPU (1.0 MB: the switch off 0), ' +
    '4 of them settled the flights in the air (0.8 MB)';
  const { summary } = parseLogLines([bufferCopies], 'buffer.log');
  assert.equal(summary.copyStats, null);
});

test('device removal: "after logged frame" uses the frame line number when the build prints it', () => {
  const removal =
    '2026-11-26T05:02:20.000000Z ERROR victory_gpu::host: host GPU device removed after a submission of 2 lists';
  const { summary, firstTs, lastTs } = parseLogLines([OLD_FRAME, presentLine(8192), stripControl(NEW_FRAME_COLOURED), removal], 'removed.log');
  assert.equal(summary.lastFrame.frame, 18432);
  assert.equal(parseExtras([OLD_FRAME, presentLine(8192), stripControl(NEW_FRAME_COLOURED), removal], firstTs, lastTs).ending.frame, 18432);
});

// The lines of job/timing-truth, in the text `frame_line`, `window_text` and `Budget::line` (victory-gpu host/stats.rs)
// write, with made-up numbers. The frame line keeps its beginning and ends with the window clause; the budget line
// follows it in a diagnostics build.
const windowFrame = ({ frame = 105, ms = 220.0, first = 101, frames = 5, wall = 1100, p50 = 219.4, p95 = 231.1, at = '2026-11-26T05:02:11.250000Z' } = {}) =>
  `${at}  INFO victory_gpu::host: frame ${frame}, ${(1000 / ms).toFixed(1)} fps, ${ms.toFixed(1)} ms/frame (12.5 preparing: copies 3.1 ms 4.2 MB, ` +
  "textures hashed 5.3 ms 6.4 MB; 7.5 host waits (for the relay 1.5, for flights' fence 2.0, synchronous submissions sent to " +
  'retired 4.0); 8.6 copying back 9.7 MB merged into title memory (host time, fetches included), 0.4 MB fetched from copies ' +
  "in the card's memory), 5000 draws/frame, dropped 3 without a host pipeline (1 being made) and 7 indirect; heap buffers " +
  '0.0 passes 0.0 ms; host waits summed across threads 7.5 ms a frame, cpu 12%; ' +
  `means over frames ${first}..${frame} (${frames} frames in ${wall} ms, p50 ${p50}, p95 ${p95} ms/frame): every frame since ` +
  "the line before; the figures in the parentheses are summed across threads and overlap, they are not parts of the frame (the frame budget line's are)";
// The budget line as the runtime's tests pin it (fixtures/runtime-lines.mjs), after a timestamp, level and module; a
// test that needs its own figures replaces them in that text, so every other character stays the runtime's.
const AT = '2026-11-26T05:02:11.251000Z  INFO victory_gpu::host: ';
const budgetLine = ({ first = 1230, last = 1234, frames = 2, frame = '200.00', phases, sum = '200.00', base = BUDGET_LINE_OTHERS } = {}) => {
  let text = base
    .replace('frames 1230..1234 (2 frames): 200.00 ms/frame', `frames ${first}..${last} (${frames} frames): ${frame} ms/frame`)
    .replace('(sum 200.00)', `(sum ${sum})`);
  if (phases) text = text.replace(/= .+? \(sum /, `= ${phases} (sum `);
  return `${AT}${text}`;
};

test('the frame line in the window shape parses as before, and its window clause gives its frames, wall time and percentiles', async () => {
  const { FRAME_RE, parseFrameWindow } = await import('./parse-run.mjs');
  const line = windowFrame();
  assert.ok(FRAME_RE.test(line), 'the beginning is unchanged');
  assert.deepEqual(parseFrameWindow(line), { firstFrame: 101, lastFrame: 105, frames: 5, wallMs: 1100, p50Ms: 219.4, p95Ms: 231.1 });
  assert.equal(parseFrameWindow(OLD_FRAME), null);
  const { summary } = parseLogLines([line], 'window.log');
  assert.equal(summary.lastFrameLine.shape, 'window');
  assert.equal(summary.lastFrameLine.hostWaitsMs, 7.5);
  assert.equal(parseLogLines([OLD_FRAME], 'old.log').summary.lastFrameLine.shape, 'older');
});

test('the frame windows are joined weighted by their frames, never as a plain mean of the lines', () => {
  // 5 frames in 1000 ms, then 15 in 4500: 275 ms a frame over the 20, where the two lines' own figures are 200 and 300.
  const lines = [
    windowFrame({ frame: 105, ms: 200, first: 101, frames: 5, wall: 1000, p50: 199, p95: 210 }),
    windowFrame({ frame: 120, ms: 300, first: 106, frames: 15, wall: 4500, p50: 290, p95: 400, at: '2026-11-26T05:02:16.250000Z' }),
  ];
  const { summary } = parseLogLines(lines, 'window.log');
  assert.deepEqual(summary.frameLines, { window: 2, older: 0 });
  assert.equal(summary.frameWindow.frames, 20);
  assert.equal(summary.frameWindow.wallMs, 5500);
  assert.equal(summary.frameWindow.meanMs, 275);
  assert.equal(summary.frameWindow.firstFrame, 101);
  assert.equal(summary.frameWindow.lastFrame, 120);
  assert.equal(summary.frameWindow.lastP95Ms, 400);
  // An older log has no window: n/a, never zero.
  const old = parseLogLines([OLD_FRAME], 'old.log').summary;
  assert.deepEqual(old.frameLines, { window: 0, older: 1 });
  assert.equal(old.frameWindow, null);
});

test("the runtime's budget line parses: its phases, the title's calls, processor times, scopes and other threads", async () => {
  const { parseBudgetLine } = await import('./parse-run.mjs');
  const budget = parseBudgetLine(`${AT}${BUDGET_LINE}`);
  assert.equal(budget.firstFrame, 1230);
  assert.equal(budget.lastFrame, 1234);
  assert.equal(budget.frames, 2);
  assert.equal(budget.frameMs, 200);
  assert.equal(budget.sumMs, 200);
  assert.deepEqual(budget.phases.map((p) => [p.name, p.ms]), [['title', 60], ['list_calls', 30], ['unaccounted', 100], ['object_calls', 6], ['hle_calls', 4]]);
  assert.equal(budget.titleAndCallsCpuMs, 90, 'the processor time of the title and its calls together');
  assert.deepEqual(budget.cpu, { unaccounted: 50 });
  assert.equal(budget.cpuMissing, false);
  assert.equal(budget.scopes, 30);
  assert.equal(budget.calls, 2000);
  assert.equal(budget.largestTitleGapMs, 3.5);
  assert.deepEqual(budget.others, [], 'nothing measured');
  assert.deepEqual(budget.unknownPhases, []);
  const apart = parseBudgetLine(BUDGET_LINE_OTHERS);
  assert.deepEqual(apart.cpu, { unaccounted: 10, relay_wait: 40 });
  assert.deepEqual(apart.others, [{ name: 'relay_land', ms: 60, cpuMs: 15 }]);
  const missing = parseBudgetLine(BUDGET_LINE.replace('of it on the processor:', "of it on the processor (the counter's rate was unknown for part of it: left out):"));
  assert.equal(missing.cpuMissing, true);
  assert.equal(parseBudgetLine(OLD_FRAME), null);
  assert.equal(parseBudgetLine(FRAME_LINE), null);
});

test("the budget's identity holds within rounding, and a budget whose phases overrun the frame or name an unknown phase is caught", async () => {
  const { budgetIdentity, parseBudgetLine } = await import('./parse-run.mjs');
  assert.equal(budgetIdentity(parseBudgetLine(BUDGET_LINE)).ok, true);
  assert.equal(budgetIdentity(parseBudgetLine(BUDGET_LINE_OTHERS)).ok, true);
  // Three phases of a third each print 33.33: 99.99 against 100.00, inside four half-units of 0.005 ms.
  const thirds = budgetIdentity(parseBudgetLine(budgetLine({ frame: '100.00', sum: '100.00', phases: 'title 33.33 + list_calls 33.33 + unaccounted 33.33' })));
  assert.equal(thirds.ok, true);
  assert.equal(thirds.toleranceMs, 0.02);
  // A nested phase counted twice (the runtime's own fixture: 272.5 ms of phases in a 161.6 ms frame).
  const nested = budgetIdentity(parseBudgetLine(budgetLine({ frame: '161.60', sum: '161.60', phases: 'title 100.00 + unaccounted 61.60 + settle 110.90' })));
  assert.equal(nested.ok, false);
  assert.equal(nested.phaseSumMs, 272.5);
  // The line's own sum against its frame.
  assert.equal(budgetIdentity(parseBudgetLine(budgetLine({ sum: '199.00' }))).ok, false);
  // A phase the runtime does not name (the first rebuilt shape's `recording`): the shape moved, so the line fails.
  const renamed = budgetIdentity(parseBudgetLine(budgetLine({ phases: 'title 100.00 + recording 20.00 + relay_wait 80.00' })));
  assert.equal(renamed.ok, false);
  assert.equal(renamed.namesOk, false);
  assert.deepEqual(renamed.unknown, ['recording']);
});

test('budget lines are joined weighted by their frames, phases a line leaves out (zero) included, and counted for the identity', () => {
  const lines = [
    windowFrame(),
    budgetLine({ first: 101, last: 105, frames: 5, frame: '200.00', sum: '200.00', phases: 'title 100.00 + list_calls 0.00 + unaccounted 20.00 + relay_wait 80.00' }),
    windowFrame({ frame: 120, first: 106, frames: 15, wall: 4500, at: '2026-11-26T05:02:16.250000Z' }),
    budgetLine({ first: 106, last: 120, frames: 15, frame: '300.00', sum: '300.00', phases: 'title 200.00 + list_calls 10.00 + unaccounted 90.00' }),
  ];
  const { summary } = parseLogLines(lines, 'budget.log');
  const b = summary.budget;
  assert.equal(b.lines, 2);
  assert.equal(b.frames, 20);
  assert.equal(b.frameMs, 275);
  assert.equal(b.phases.title, 175);
  assert.equal(b.phases.relay_wait, 20);
  assert.equal(b.phases.unaccounted, 72.5);
  assert.equal(b.phases.list_calls, 7.5);
  assert.equal(b.identity.failedLines, 0);
  assert.equal(b.identity.checkedLines, 2);
  assert.deepEqual(b.identity.unknownPhases, []);
  assert.equal(b.calls, 0);
  assert.equal(b.largestTitleGapMs, 0);
  assert.deepEqual(Object.keys(b.phases), ['title', 'list_calls', 'unaccounted', 'relay_wait'], "in the runtime's order");
  assert.equal(parseLogLines([OLD_FRAME], 'old.log').summary.budget, null);
});
