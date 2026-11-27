import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { isolatedEnv } from '../test-isolation.mjs';
import { writeAtomic } from '../live.mjs';
import { DEFAULT_START, listedLive, runtimeParse, segmentLine as runtimeSegmentLine, virtualClock } from '../live-fake-runtime.mjs';
import { createProtocol, parseArgs } from './live-server.mjs';
import { LIVE_NAMES } from '../live-names.mjs';
import { sampleObject } from '../fixtures/old-shape.mjs';
import { BUDGET_LINE } from '../fixtures/runtime-lines.mjs';
import { RUNTIME_PHASES, statsObject } from '../fixtures/runtime-stats.mjs';

// The query mark of a web address, joined in at run time so that no address literal here carries one.
const Q = '?';

/// The runtime's object (fixtures/runtime-stats.mjs) with a test's own fields; a name the runtime does not write is
/// refused.
const intervalObject = (overrides = {}) => statsObject(overrides);

const here = path.dirname(fileURLToPath(import.meta.url));
const server = path.join(here, 'live-server.mjs');
const repo = path.resolve(here, '..', '..');
const TOOL_NAMES = [
  'game_status',
  'frame_parts',
  'live_get',
  'live_set',
  'segments',
  'run_plan',
  'send_keys',
  'log_find',
];

function tempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'live-mcp-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

const FRAME_PREFIX = '2026-11-26T10:00:';

/// One runtime frame line, with every cost field COST_RE names.
function frameLine(second, frame, fps, ms, copiesMs = 0.1, copiesMb = 0.2) {
  const stamp = `${FRAME_PREFIX}${String(second).padStart(2, '0')}Z`;
  return (
    `${stamp}  INFO victory_gpu::host: frame ${frame}, ${fps} fps, ${ms} ms/frame ` +
    `(0.2 preparing: copies ${copiesMs} ms ${copiesMb} MB, textures hashed 0.0 ms 0.0 MB; ` +
    `0.9 on the GPU; 0.0 copying back 0.0 MB), 12 draws/frame, dropped 0 without a host pipeline ` +
    '(0 being made) and 1 indirect'
  );
}

/// The pre-rebuild segment line (every present counted), kept for reading old logs.
function segmentLine(number, first, last, mean, inForce = {}) {
  const forced = Object.entries(inForce)
    .map(([name, value]) => `${name}=${value}`)
    .join(' ');
  return (
    `live segment ${number}: frames ${first}..${last}, mean ${mean.toFixed(1)}, p50 ${mean.toFixed(1)}, ` +
    `p95 ${(mean + 1.2).toFixed(1)}, skipped 0 draws and 0 dispatches, in force: ${forced}`
  );
}

function cannedLog(dir, { extra = [] } = {}) {
  const file = path.join(dir, 'run.log');
  const lines = [
    'harness build: abc1234 profile=diag console=series-x',
    'harness checkout: 0123456789abcdef0123456789abcdef01234567',
    'harness env: VICTORY_LOG=info',
    'harness env: VICTORY_WAVE64_EMULATE=emulate',
    'harness env: VICTORY_GPU_TIMING=1',
    'harness env: VICTORY_CAPTURE=set, outside the repository',
    frameLine(0, 100, 60.0, 16.7, 0.1, 0.2),
    frameLine(9, 160, 60.2, 16.6, 0.2, 0.4),
    frameLine(10, 180, 59.9, 16.8, 0.3, 0.6),
    segmentLine(1, 1, 150, 16.7, {
      VICTORY_KEEP_TEXTURES: 'on',
      VICTORY_TEXTURE_LOW: '70',
      VICTORY_TEXTURE_HIGH: '80',
      LIVE_SKIP_DISPATCH: '',
      LIVE_SKIP_DRAW: '',
    }),
    'live: VICTORY_KEEP_TEXTURES off -> on at frame 151',
    ...extra,
  ];
  fs.writeFileSync(file, `${lines.join('\n')}\n`);
  return file;
}

function cannedStats(dir, data = null) {
  const file = path.join(dir, 'stats.json');
  fs.writeFileSync(
    file,
    `${JSON.stringify(
      data ?? {
        frame: 181,
        fps: 59.9,
        frame_ms: 16.8,
        parts: { copies_ms: 1.5, copies_mb: 2.25, textures: { ms: 0.5, mb: 1.0 } },
        draws: 12,
        vram_mb: 18_000,
        live: { VICTORY_KEEP_TEXTURES: 'on', VICTORY_TEXTURE_LOW: '70' },
        segment: 2,
        custom_counter: 7,
        pid: 4242,
        command_line: 'a secret command',
        path: ['Z:', 'made', 'up'].join('/'),
      },
    )}\n`,
  );
  return file;
}

// The private keys a stats object must never pass through, whatever its shape.
const PRIVATE_KEYS = { custom_counter: 7, pid: 4242, command_line: 'a secret command', path: ['Z:', 'made', 'up'].join('/') };

/// The runtime's object (fixtures/runtime-stats.mjs) written now, with the private keys beside it.
function cannedIntervalStats(dir, overrides = {}) {
  return cannedStats(dir, { ...intervalObject({ unix_ms: Date.now(), ...overrides }), ...PRIVATE_KEYS });
}

/// The runtime's segment line (live.rs `segment_line`, written by the fake's writer, which gives the fixtures for the
/// runtime's own tests), closed or, with `ended: null`, the open one under way.
function countedSegmentLine({
  number, first, last, counted, excluded = 3, excludedMean = 50.0, mean, p50 = mean, p95 = mean + 1.2, light = 0,
  lightMean = 120.0, long = 0, longMean = 1500.0, begun = 'change', ended = 'change', inForce = '',
}) {
  const figures = {
    counted,
    transition: [excluded, excluded ? excludedMean : 0],
    light: [light, light ? lightMean : 0],
    long: [long, long ? longMean : 0],
    mean,
    p50,
    p95,
  };
  return runtimeSegmentLine({ k: number, first, last, figures, begun, ended, inForce });
}

/// The frame line from job/timing-truth on: its figures are means over every frame since the line before, said in the
/// closing clause, and the parenthesised ones are summed across threads.
function windowFrameLine(stamp, { frame, fps, ms, first, frames, wallMs, p50, p95 }) {
  return (
    `${stamp}  INFO victory_gpu::host: frame ${frame}, ${fps} fps, ${ms} ms/frame (2.25 preparing: copies 1.75 ms ` +
    "3.5 MB, textures hashed 0.5 ms 0.25 MB; 6.5 host waits (for the relay 1.5, for flights' fence 2.0, synchronous " +
    'submissions sent to retired 3.0); 0.75 copying back 1.25 MB merged into title memory (host time, fetches ' +
    "included), 0.5 MB fetched from copies in the card's memory), 6123 draws/frame, dropped 4 without a host pipeline " +
    '(0 being made) and 2 indirect; host waits summed across threads 6.5 ms a frame, cpu 12%; means over frames ' +
    `${first}..${frame} (${frames} frames in ${wallMs} ms, p50 ${p50}, p95 ${p95} ms/frame): every frame since the ` +
    'line before; the figures in the parentheses are summed across threads and overlap, they are not parts of the ' +
    "frame (the frame budget line's are)"
  );
}

/// The budget line (diagnostics builds), right after a frame line.
function budgetLine(stamp, { first, last, frames, frameMs, phases }) {
  const sum = Object.values(phases).reduce((total, value) => total + value, 0);
  const parts = Object.entries(phases)
    .map(([name, value]) => `${name} ${value.toFixed(2)}`)
    .join(' + ');
  const text = BUDGET_LINE.replace('frames 1230..1234 (2 frames): 200.00 ms/frame', `frames ${first}..${last} (${frames} frames): ${frameMs.toFixed(2)} ms/frame`)
    .replace(/= .+? \(sum 200\.00\)/, `= ${parts} (sum ${sum.toFixed(2)})`)
    .replace('the title and its calls together 90.00, unaccounted 50.00;', 'the title and its calls together 19.50;');
  return `${stamp}  INFO victory_gpu::host: ${text}`;
}

/// Every budget phase present (the runtime writes them all), zero unless given.
function fullPhases(given) {
  const presenting = RUNTIME_PHASES.filter((name) => name !== 'pool_work' && name !== 'relay_land');
  const entry = ['title', 'list_calls', 'object_calls', 'hle_calls'];
  return Object.fromEntries(presenting.map((name) => [name, given[name] ?? { ms: 0, cpu_ms: entry.includes(name) ? null : 0 }]));
}

// The known input for an independent check of every tool: one interval-shape stats object and one run log, each
// field a distinct value, so a tool's answer can be checked field by field. The budget's phases sum to its frame.
function knownIntervalObject(now = Date.now()) {
  return statsObject({
    seq: 731,
    t_ms: 98765.4,
    unix_ms: now - 2000,
    interval: { first_frame: 5001, last_frame: 5004, frames: 4, wall_ms: 140.0 },
    frame: 5004,
    fps: 28.57,
    frame_ms: 35.0,
    frame_ms_p50: 34.1,
    frame_ms_worst: 38.7,
    frame_ms_list: [33.2, 34.1, 38.7, 34.0],
    frame_ms_list_missing: 0,
    budget: {
      thread: 'presenting',
      frames: 4,
      frame_ms: 35.0,
      sum_ms: 35.0,
      scopes: 212.0,
      calls: 3100.0,
      largest_title_gap_ms: 2.5,
      title_and_calls_cpu_ms: 20.0,
      phases: fullPhases({
        title: { ms: 21.25, cpu_ms: null },
        list_calls: { ms: 3.125, cpu_ms: null },
        unaccounted: { ms: 1.375, cpu_ms: 0.5 },
        relay_wait: { ms: 9.25, cpu_ms: 0.75 },
      }),
    },
    other_threads: { scopes: 44.0, phases: { list_calls: { ms: 4.5, cpu_ms: null } } },
    summed_across_threads: { preparing_ms: 2.25, copies_ms: 1.75, host_wait_ms: 6.5, fetch_queue_fallback_by: { numbered: 0.5 } },
    gpu_stamps: { copy_queue: { fetch_lists: 2.5, stamped: 2.0, ms: 1.125, pairs_unread: 0.5 }, not_stamped: [], idle: 'not derivable' },
    draws: 6123.0,
    dispatched: 287.0,
    dropped: 4.5,
    stores_failed_total: 3,
    stores_failed_bytes_total: 4096,
    pipelines_created_total: 5321,
    pipelines_per_minute: 1.5,
    vram_mb: 17777,
    live: { VICTORY_KEEP_TEXTURES: 'off', VICTORY_TEXTURE_LOW: '61', VICTORY_TEXTURE_HIGH: '85', VICTORY_LATE_WAITS: 'on' },
    segments_closed_total: 6,
    segment_open: {
      segment: 7, first_frame: 4950, last_frame: 5004, counted: 52, transition_excluded: 3, transition_mean_ms: 61.5,
      light_excluded: 1, light_mean_ms: 120.0, long_excluded: 0, long_mean_ms: 0.0, mean_ms: 35.5, p50_ms: 34.6,
      p95_ms: 39.2, begun: 'change',
    },
    // No optional part: every switch that adds one is off.
    kernel_calls: { per_frame: { reads: 1.5 }, reads_total: 600 },
  });
}

const KNOWN_IN_FORCE_5 = 'VICTORY_KEEP_TEXTURES=on VICTORY_TEXTURE_LOW=70 VICTORY_TEXTURE_HIGH=85';
const KNOWN_IN_FORCE_6 = 'VICTORY_KEEP_TEXTURES=on VICTORY_TEXTURE_LOW=64 VICTORY_TEXTURE_HIGH=85';
const KNOWN_IN_FORCE_7 = 'VICTORY_KEEP_TEXTURES=off VICTORY_TEXTURE_LOW=61 VICTORY_TEXTURE_HIGH=85';

function knownLog(dir) {
  const file = path.join(dir, 'run.log');
  const lines = [
    'harness build: abc1234 profile=diag console=series-x',
    'harness checkout: 0123456789abcdef0123456789abcdef01234567',
    'harness env: VICTORY_LOG=info',
    windowFrameLine(`${FRAME_PREFIX}01Z`, { frame: 4900, fps: 27.78, ms: 36.0, first: 4865, frames: 36, wallMs: 1296, p50: 35.5, p95: 40.5 }),
    budgetLine(`${FRAME_PREFIX}01Z`, { first: 4865, last: 4900, frames: 36, frameMs: 36.0, phases: { title: 22.0, list_calls: 3.0, unaccounted: 1.0, relay_wait: 10.0 } }),
    countedSegmentLine({ number: 5, first: 4000, last: 4499, counted: 497, excludedMean: 88.5, mean: 36.2, p50: 35.9, p95: 40.4, light: 2, inForce: KNOWN_IN_FORCE_5 }),
    'live: VICTORY_TEXTURE_LOW 70 -> 64 at frame 4500',
    countedSegmentLine({ number: 6, first: 4500, last: 4949, counted: 447, excludedMean: 77.0, mean: 35.8, p50: 35.1, p95: 39.9, inForce: KNOWN_IN_FORCE_6 }),
    'live: VICTORY_KEEP_TEXTURES on -> off at frame 4950',
    'live: VICTORY_TEXTURE_LOW 64 -> 61 at frame 4950',
    windowFrameLine(`${FRAME_PREFIX}07Z`, { frame: 5004, fps: 28.57, ms: 35.0, first: 5001, frames: 4, wallMs: 140, p50: 34.1, p95: 38.7 }),
    budgetLine(`${FRAME_PREFIX}07Z`, { first: 5001, last: 5004, frames: 4, frameMs: 35.0, phases: { title: 21.25, list_calls: 3.13, unaccounted: 1.37, relay_wait: 9.25 } }),
    countedSegmentLine({ number: 7, first: 4950, last: 5004, counted: 52, excludedMean: 61.5, mean: 35.5, p50: 34.6, p95: 39.2, light: 1, ended: null, inForce: KNOWN_IN_FORCE_7 }),
    `${FRAME_PREFIX}08Z  WARN victory_gpu::host: a known warning for log_find, shader deadbeefcafebabe0 skipped`,
  ];
  fs.writeFileSync(file, `${lines.join('\n')}\n`);
  return file;
}

function writeControl(dir, text) {
  const file = path.join(dir, 'live.txt');
  fs.writeFileSync(file, text);
  return file;
}

function makeLock(dir, started = new Date(Date.now() - 42_000)) {
  const lock = path.join(dir, 'run.lock');
  fs.mkdirSync(lock);
  const stamp = started.toISOString().slice(0, 19).replace('T', ' ');
  // The owner line is private data the server must never repeat: a pid, a folder and a command line.
  fs.writeFileSync(path.join(lock, 'owner'), `pid 4242, started ${stamp}, VICTORY_DIR=/made/up/lane, args: --live\n`);
  return lock;
}

/// The server under test. By default it runs in this process on a virtual clock (`clock`, else a new one), so that
/// no age, wait or timeout in a tool depends on the machine's speed: a fake game given the same clock moves only as
/// the server's waits move the clock. `child: true` runs the real stdio process instead, for the tests of the
/// protocol, the arguments and the process; those wait on its output with a generous bound and assert on order.
function startServer(t, { args = [], env = {}, clock = null, child: asChild = false } = {}) {
  if (!asChild) return startInProcess(t, { args, env, clock: clock ?? virtualClock() });
  const child = spawn(process.execPath, [server, ...args], {
    cwd: repo,
    env: isolatedEnv({ ...env }),
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  let buffer = '';
  let stderr = '';
  const lines = [];
  const nonJson = [];
  const waiting = new Map();
  let nextId = 1;

  child.stderr.on('data', (chunk) => {
    stderr += chunk;
  });
  child.stdout.on('data', (chunk) => {
    buffer += chunk;
    for (;;) {
      const newline = buffer.indexOf('\n');
      if (newline < 0) break;
      const line = buffer.slice(0, newline);
      buffer = buffer.slice(newline + 1);
      if (!line.trim()) continue;
      let message;
      try {
        message = JSON.parse(line);
      } catch {
        nonJson.push(line);
        continue;
      }
      if (message.jsonrpc !== '2.0') nonJson.push(line);
      lines.push(message);
      if (message.id !== undefined && message.id !== null && waiting.has(message.id)) {
        waiting.get(message.id)(message);
        waiting.delete(message.id);
      }
    }
  });

  function request(method, params) {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        waiting.delete(id);
        reject(new Error(`timed out waiting for ${method}; stderr: ${stderr}`));
      }, 120_000);
      waiting.set(id, (message) => {
        clearTimeout(timer);
        resolve(message);
      });
      child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
    });
  }

  function notify(method, params) {
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method, params })}\n`);
  }

  async function callTool(name, args = {}) {
    const message = await request('tools/call', { name, arguments: args });
    assert.ok(message.result, `tools/call ${name} has a result: ${JSON.stringify(message)}`);
    const block = message.result.content[0];
    assert.equal(block.type, 'text');
    return { value: JSON.parse(block.text), isError: message.result.isError, result: message.result };
  }

  async function stop() {
    if (child.exitCode !== null) return;
    child.stdin.end();
    await new Promise((resolve) => {
      const timer = setTimeout(() => {
        child.kill();
        resolve();
      }, 60_000);
      child.once('exit', () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }

  t.after(stop);
  return { child, request, notify, callTool, stop, lines, nonJson, stderr: () => stderr };
}

function startInProcess(t, { args, env, clock }) {
  const isolated = isolatedEnv({ ...env });
  let said = '';
  const protocol = createProtocol({ env: isolated, out: { write: (text) => { said += text; } }, clock });
  protocol.setRunFiles(parseArgs(args, isolated));
  let nextId = 1;
  const lines = [];
  async function request(method, params) {
    const answer = await protocol.handle({ jsonrpc: '2.0', id: nextId++, method, params });
    // As the stdio process would send it: through JSON.
    const message = answer === null ? null : JSON.parse(JSON.stringify(answer));
    if (message) lines.push(message);
    return message;
  }
  async function callTool(name, args2 = {}) {
    const message = await request('tools/call', { name, arguments: args2 });
    assert.ok(message.result, `tools/call ${name} has a result: ${JSON.stringify(message)}`);
    const block = message.result.content[0];
    assert.equal(block.type, 'text');
    return { value: JSON.parse(block.text), isError: message.result.isError, result: message.result };
  }
  const stop = async () => protocol.close();
  t.after(stop);
  return { clock, request, callTool, stop, lines, nonJson: [], stderr: () => said };
}

/// A stand-in for the runtime, on live.rs's rule: only the names present in the control file are applied (a removed
/// name keeps its value), and a change logs the segment it closes and then a marker per name. It writes the stats
/// file a --live run has (stats.json beside the log), its frame advancing every tick. It never touches the game.
/// `shape` 'interval' writes job/timing-truth's shape: a stats object with seq, its own clock and the frames since the
/// last one, the counted segment line, and the open line of the segment under way; 'sample' the older build's. A
/// LIVE_KEYS change is answered with the runtime's done line.
function fakeRuntime(t, { log, control, segments = true, shape = 'sample', clock = virtualClock() }) {
  const assignments = () => {
    let text = '';
    try {
      text = fs.readFileSync(control, 'utf8');
    } catch {
      // No file yet.
    }
    return runtimeParse(text) ?? [];
  };
  let state = { ...DEFAULT_START };
  for (const [name, value] of assignments()) state[name] = value;
  let frame = 100;
  let statsFrame = 1;
  let segment = 0;
  let seq = 0;
  const stats = path.join(path.dirname(log), 'stats.json');
  const inForceText = () => LIVE_NAMES.map((name) => `${name}=${state[name]}`).join(' ');
  const writeStats = () => {
    statsFrame += 10;
    seq += 1;
    const object =
      shape === 'interval'
        ? intervalObject({
            seq,
            t_ms: seq * 200,
            unix_ms: clock.now(),
            interval: { first_frame: statsFrame - 9, last_frame: statsFrame, frames: 10, wall_ms: 200 },
            frame: statsFrame,
            fps: 50,
            frame_ms: 20,
            frame_ms_p50: 20,
            frame_ms_worst: 20,
            frame_ms_list: Array.from({ length: 10 }, () => 20),
            live: listedLive(state),
            segments_closed_total: segment,
            segment_open: null,
          })
        : { frame: statsFrame, fps: 50, frame_ms: 20, live: listedLive(state), segment };
    writeAtomic(stats, `${JSON.stringify(object)}\n`);
    if (shape === 'interval' && segments) {
      const open = { number: segment + 1, first: frame, last: frame + 5, counted: 3, mean: 20.0, ended: null };
      fs.appendFileSync(log, `${countedSegmentLine({ ...open, inForce: inForceText() })}\n`);
    }
  };
  writeStats();
  const cancel = clock.every(20, () => {
    writeStats();
    const changed = assignments().filter(([name, value]) => state[name] !== value);
    if (!changed.length) return;
    if (segments) {
      segment += 1;
      fs.appendFileSync(
        log,
        shape === 'interval'
          ? `${countedSegmentLine({ number: segment, first: frame, last: frame + 120, counted: 118, mean: 20.0, inForce: inForceText() })}\n`
          : `${segmentLine(segment, frame, frame + 120, 20.0, {})} ${inForceText()}\n`,
      );
      frame += 121;
    }
    for (const [name, value] of changed) {
      fs.appendFileSync(log, `live: ${name} ${state[name]} -> ${value} at frame ${frame}\n`);
      state = { ...state, [name]: value };
      if (name === 'LIVE_KEYS' && value) {
        const presses = value.split(',').filter((entry) => entry.trim()).length;
        fs.appendFileSync(log, `live keys: sequence done at frame ${frame + 30}, ${presses} presses\n`);
      }
    }
  });
  t.after(cancel);
  return { stop: cancel, clock };
}

test('initialize echoes the protocol version, advertises tools, and the initialized notification is silent', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const s = startServer(t, { child: true, args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });

  const init = await s.request('initialize', { protocolVersion: '2024-11-05' });
  assert.equal(init.result.protocolVersion, '2024-11-05');
  assert.deepEqual(init.result.capabilities, { tools: {} });
  assert.equal(init.result.serverInfo.name, 'victory-live');
  assert.match(init.result.instructions, /starts, ends or kills a run/);

  s.notify('notifications/initialized');
  const before = s.lines.length;
  const ping = await s.request('ping');
  assert.deepEqual(ping.result, {});
  // The server answers its input in order, so anything said to the notification came before the ping's answer.
  assert.equal(s.lines.length, before + 1, 'the notification got no answer');
  assert.deepEqual(s.nonJson, [], 'only JSON-RPC reached standard output');
});

test('tools/list has the eight tools and a valid object schema for each', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const s = startServer(t, { child: true, args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });

  const listed = await s.request('tools/list');
  const tools = listed.result.tools;
  assert.deepEqual(
    tools.map((tool) => tool.name).sort(),
    [...TOOL_NAMES].sort(),
  );
  for (const tool of tools) {
    assert.equal(typeof tool.description, 'string', `${tool.name} has a description`);
    assert.ok(tool.description.length > 20, `${tool.name} describes when to use it`);
    assert.equal(tool.inputSchema.type, 'object', `${tool.name} takes an object`);
    assert.equal(typeof tool.inputSchema.properties, 'object', `${tool.name} lists properties`);
    if (tool.inputSchema.required) assert.ok(Array.isArray(tool.inputSchema.required));
  }
  assert.deepEqual(s.nonJson, []);
});

test('game_status reads the header, lock and stats and never repeats a path, pid or command line', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const stats = cannedStats(dir);
  const control = writeControl(dir, 'VICTORY_KEEP_TEXTURES=on\n');
  const made = Date.now();
  const lock = makeLock(dir);
  const s = startServer(t, {
    args: ['--log', log, '--file', control, '--stats', stats, '--lock', lock],
  });

  const { value, isError } = await s.callTool('game_status');
  assert.equal(isError, false);
  assert.equal(value.running, true);
  assert.equal(value.locked, true);
  assert.equal(value.build, 'abc1234');
  assert.equal(value.profile, 'diag');
  assert.equal(value.console, 'series-x');
  assert.equal(value.checkout, '0123456789abcdef0123456789abcdef01234567');
  assert.equal(value.switches.VICTORY_WAVE64_EMULATE, 'emulate');
  assert.equal(value.switches.VICTORY_GPU_TIMING, '1');
  // The lock's start is 42 s before it was made, to the second; the upper bound is however long this machine took.
  const most = 43 + Math.ceil((Date.now() - made) / 1000);
  assert.ok(value.uptime_s >= 41 && value.uptime_s <= most, `uptime was ${value.uptime_s}, not 41..${most}`);
  assert.equal(value.fps, 59.9);
  assert.equal(value.frame_ms, 16.8);
  assert.equal(value.vram_mb, 18_000);
  // Corrected (was value.parts.copies_ms 1.5): the older shape's `parts` are summed across threads and overlap; they
  // are not parts of the frame, and an older build has no budget to give those.
  assert.equal(value.summed_across_threads.copies_ms, 1.5);
  assert.equal(value.parts, null, 'n/a: no budget in the older shape');
  assert.equal(value.stats_shape, 'sample');
  assert.equal(value.seq, null);
  assert.equal(value.stats_age_by, 'file_time', 'the older shape has no clock of its own');
  assert.equal(value.stats_stale, false);
  assert.equal(value.frame_ms_p50, null, 'n/a, never zero');
  assert.equal(value.dispatched, null, 'n/a: the older shape counted dispatches since the previous object');
  assert.equal(value.segment_open, null);
  assert.equal(value.source, 'stats');
  assert.equal(value.stats.custom_counter, 7, 'an unknown stats field passes through');
  assert.equal('pid' in value.stats, false, 'a process id is dropped');
  assert.equal('command_line' in value.stats, false, 'a command line is dropped');
  assert.equal('path' in value.stats, false, 'a path field is dropped');
  const text = JSON.stringify(value);
  assert.equal(text.includes(dir), false, 'no folder path');
  assert.equal(text.includes('4242'), false, 'no process id');
  assert.equal(text.includes('a secret command'), false, 'no command line');
  assert.equal(text.includes('args:'), false, 'no command line');
  assert.equal(/[A-Za-z]:[\\/]/.test(text), false, 'no drive path');
});

test('game_status falls back to the log frame line when there is no stats file', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const s = startServer(t, { args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });

  const { value } = await s.callTool('game_status');
  assert.equal(value.running, false);
  assert.equal(value.source, 'log');
  assert.equal(value.fps, 59.9);
  assert.equal(value.frame_ms, 16.8);
  // Corrected (was value.parts.copies_ms 0.3): the frame line's parenthesised figures are summed across threads.
  assert.equal(value.summed_across_threads.copies_ms, 0.3);
  assert.equal(value.parts, null, 'no budget line in this log');
  assert.equal(value.frame_line_shape, 'sample', 'an older frame line: no window clause');
  assert.equal(value.stats_shape, null);
  assert.equal(value.vram_mb, null);
  assert.equal(value.live.VICTORY_KEEP_TEXTURES, 'on');
});

test('game_status reads an interval-shape object: its shape, seq, age by its own clock, open segment and budget', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const clock = virtualClock();
  const stats = cannedIntervalStats(dir, { unix_ms: clock.now() - 3000 });
  const s = startServer(t, { clock, args: ['--log', log, '--stats', stats, '--lock', path.join(dir, 'run.lock')] });

  const { value, isError } = await s.callTool('game_status');
  assert.equal(isError, false);
  assert.equal(value.source, 'stats');
  assert.equal(value.stats_shape, 'interval');
  // The runtime's object (fixtures/runtime-stats.mjs STATS_LIVE), its figures read by their names.
  assert.equal(value.seq, 8);
  assert.equal(value.stats_age_by, 'unix_ms');
  assert.equal(value.stats_age_s, 3, 'age by the object\'s own clock');
  assert.equal(value.stats_stale, false);
  assert.equal(value.frame_ms, 260);
  assert.equal(value.frame_ms_p50, 220);
  assert.equal(value.frame_ms_worst, 420, 'the object\'s longest frame');
  assert.equal('frame_ms_p95' in value, false, 'an object has no 95th percentile: not offered as a null');
  assert.deepEqual(value.interval, { first_frame: 1230, last_frame: 1234, frames: 5, wall_ms: 1300 });
  assert.equal(value.dispatched, 12, 'per frame');
  assert.equal(value.dispatches_since_previous, null);
  assert.equal(value.summed_across_threads.host_wait_ms, 78);
  assert.deepEqual(value.parts.title, { ms: 60, cpu_ms: null });
  assert.deepEqual(value.parts.unaccounted, { ms: 100, cpu_ms: 50 });
  assert.equal(value.segments_closed_total, 1);
  assert.equal('segment' in value, false, 'the closed count goes by the runtime\'s name');
  assert.deepEqual(value.segment_open, {
    segment: 2, first_frame: 10, last_frame: 19, counted: 4, transition_excluded: 3, transition_mean_ms: 310,
    light_excluded: 1, light_mean_ms: 90, long_excluded: 2, long_mean_ms: 1500, mean_ms: 25, p50_ms: 20, p95_ms: 40,
    begun: 'change', source: 'stats',
  });
  assert.deepEqual(value.absent_optional, ['hash_check', 'texture_receipts', 'landed_classes', 'texture_recheck', 'textures_made']);
  assert.equal('pid' in value.stats, false, 'a process id is dropped in this shape too');
  assert.equal(JSON.stringify(value).includes('a secret command'), false);
});

test('game_status reports a stale object as stale, by its own clock, and withholds its figures', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  // Written a minute ago by its own clock, though the file is new (a copy, or a runtime that stopped presenting).
  const clock = virtualClock();
  const stats = cannedIntervalStats(dir, { unix_ms: clock.now() - 60_000 });
  const s = startServer(t, { clock, args: ['--log', log, '--stats', stats, '--lock', path.join(dir, 'run.lock')] });

  const { value } = await s.callTool('game_status');
  assert.equal(value.stats_stale, true);
  assert.equal(value.stats_age_s, 60, 'age by the object\'s own clock');
  assert.equal(value.fps, null, 'a stale object\'s rate is not served as current');
  assert.equal(value.frame_ms, null);
  assert.equal(value.parts, null);
  assert.equal(value.summed_across_threads, null);
  assert.match(value.stale_note, /not current/);
  assert.equal(value.stats.seq, 8, 'the last object is still there to read, as the last one');
  const parts = await s.callTool('frame_parts', { seconds: 10 });
  assert.equal(parts.value.stats_stale, true);
  assert.equal(parts.value.samples, 0);
  assert.equal(parts.value.frame_ms, null, 'and frame_parts does not serve it either');
  assert.equal(parts.value.source, 'stats', 'nor falls back to an equally old log line');
});

test('game_status calls one seq standing past the limit stale, even with a fresh clock', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const clock = virtualClock();
  // Its own clock reads 1.6 s ahead, so at the second read its age is exactly 0: only the standing seq can make it stale.
  const stats = cannedIntervalStats(dir, { unix_ms: clock.now() + 1600 });
  const s = startServer(t, {
    clock,
    args: ['--log', log, '--stats', stats, '--lock', path.join(dir, 'run.lock')],
    env: { HARNESS_MCP_STALE_S: '1' },
  });
  assert.equal((await s.callTool('game_status')).value.stats_stale, false);
  await clock.sleep(1600);
  // The same seq written again with a new clock: a writer that repeats itself, not a new object.
  cannedIntervalStats(dir, { unix_ms: clock.now() });
  const { value } = await s.callTool('game_status');
  assert.equal(value.seq, 8);
  assert.equal(value.stats_age_s, 0, 'by its own clock it is fresh');
  assert.equal(value.seq_unchanged_s, 1.6, `seq stood ${value.seq_unchanged_s} s`);
  assert.equal(value.stats_stale, true);
});

test('a stats file that appears after the server starts is picked up', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const s = startServer(t, { args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });

  const before = await s.callTool('game_status');
  assert.equal(before.value.source, 'log');
  cannedStats(dir);
  const after = await s.callTool('game_status');
  assert.equal(after.value.source, 'stats');
  assert.equal(after.value.fps, 59.9);
  const parts = await s.callTool('frame_parts', { seconds: 10 });
  assert.equal(parts.value.source, 'stats');
  assert.equal(parts.value.frame_ms.mean, 16.8);
});

test('frame_parts summarizes the stats file, parts and nested parts included', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const stats = cannedStats(dir);
  const s = startServer(t, {
    args: ['--log', log, '--stats', stats, '--lock', path.join(dir, 'run.lock')],
  });

  const { value, isError } = await s.callTool('frame_parts', { seconds: 10 });
  assert.equal(isError, false);
  assert.equal(value.source, 'stats');
  assert.equal(value.stats_shape, 'sample');
  assert.equal(value.frame_ms.mean, 16.8);
  assert.equal(value.frame_ms.median, 16.8);
  assert.equal(value.frame_ms.p95, 16.8);
  // Corrected (was value.parts.*): the older shape's `parts` are summed across threads, not parts of the frame.
  assert.equal(value.summed_across_threads.copies_ms.mean, 1.5);
  assert.equal(value.summed_across_threads.copies_mb.mean, 2.25);
  assert.equal(value.summed_across_threads['textures.ms'].mean, 0.5);
  assert.equal(value.summed_across_threads['textures.mb'].mean, 1.0);
  assert.match(value.summed_note, /overlap/);
  assert.equal(value.parts, null, 'n/a: no budget in the older shape');
  assert.match(value.parts_note, /n\/a/);
});

test('frame_parts on one interval object: its own p50 and p95, the budget phases as the parts, and the identity', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const stats = cannedIntervalStats(dir);
  const s = startServer(t, { args: ['--log', log, '--stats', stats, '--lock', path.join(dir, 'run.lock')] });

  const { value } = await s.callTool('frame_parts', { seconds: 10 });
  assert.equal(value.source, 'stats');
  assert.equal(value.stats_shape, 'interval');
  assert.equal(value.samples, 1);
  assert.equal(value.frames, 5);
  assert.equal(value.frame_ms.mean, 260);
  assert.equal(value.frame_ms.median, 220, 'the object\'s own p50, over all its frames');
  assert.equal(value.frame_ms.p95, 420, 'nearest rank of its frame list: the object itself has no p95');
  assert.equal(value.parts.title.mean, 60);
  assert.equal(value.parts.title.cpu_ms, null, 'the title has no processor time of its own');
  assert.equal(value.parts.list_calls.cpu_ms, null);
  assert.equal(value.summed_across_threads.preparing_ms.mean, 96);
  assert.equal(value.budget.identity, 'holds', "the runtime's phases sum to its frame");
});

test('frame_parts joins interval objects by their frames, skips a repeated seq, counts a jump, and checks the identity', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const now = Date.now();
  // A pause is one object holding the whole gap: one 1000 ms frame. Then nine frames of 100 ms.
  const pause = intervalObject({
    seq: 10, unix_ms: now - 3000, frame: 501, interval: { first_frame: 501, last_frame: 501, frames: 1, wall_ms: 1000 },
    frame_ms: 1000, frame_ms_p50: 1000, frame_ms_worst: 1000, frame_ms_list: [1000],
    budget: { thread: 'presenting', frames: 1, frame_ms: 1000, sum_ms: 1000, scopes: 9, phases: fullPhases({ title: { ms: 1000, cpu_ms: null } }) },
    summed_across_threads: { preparing_ms: 10 },
  });
  const steady = intervalObject({
    seq: 13, unix_ms: now - 1000, frame: 510, interval: { first_frame: 502, last_frame: 510, frames: 9, wall_ms: 900 },
    frame_ms: 100, frame_ms_p50: 100, frame_ms_worst: 100, frame_ms_list: Array.from({ length: 9 }, () => 100),
    budget: { thread: 'presenting', frames: 9, frame_ms: 100, sum_ms: 100, scopes: 9, phases: fullPhases({ title: { ms: 60, cpu_ms: null }, unaccounted: { ms: 40, cpu_ms: 1 } }) },
    summed_across_threads: { preparing_ms: 1 },
  });
  const stats = cannedStats(dir, pause);
  const s = startServer(t, { args: ['--log', log, '--stats', stats, '--lock', path.join(dir, 'run.lock')] });
  await s.callTool('game_status');
  // The same object written again, its file time moved: one object, not two.
  cannedStats(dir, pause);
  fs.utimesSync(stats, new Date(), new Date(Date.now() + 5000));
  await s.callTool('game_status');
  cannedStats(dir, steady);

  const { value } = await s.callTool('frame_parts', { seconds: 10 });
  assert.equal(value.samples, 2, 'a repeated seq is not a new sample');
  assert.equal(value.objects_missed, 2, 'seq 11 and 12 were written and never read');
  assert.equal(value.frames, 10);
  assert.equal(value.frame_ms.mean, 190, '(1000 + 9 x 100) / 10 frames, not the mean of two objects');
  assert.equal(value.frame_ms.median, 100, 'from the joined frame lists');
  assert.equal(value.frame_ms.p95, 1000);
  assert.equal(value.parts.title.mean, 154, '(1000 x 1 + 60 x 9) / 10');
  assert.equal(value.parts.unaccounted.mean, 36, '(0 x 1 + 40 x 9) / 10');
  assert.equal(value.budget.identity, 'holds');
  assert.equal(value.summed_across_threads.preparing_ms.mean, 1.9, '(10 x 1 + 1 x 9) / 10');
});

test('frame_parts falls back to the log frame lines and keeps the requested window', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const s = startServer(t, { args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });

  const { value } = await s.callTool('frame_parts', { seconds: 5 });
  assert.equal(value.source, 'log');
  assert.equal(value.samples, 2, 'the 10 s old frame is outside a 5 s window');
  assert.equal(value.frame_ms.mean, 16.7);
  // Corrected (was value.parts.copies_ms.mean 0.25): the frame line's figures are summed across threads.
  assert.equal(value.summed_across_threads.copies_ms.mean, 0.25);
  assert.equal(value.parts, null);
  assert.equal(value.frame_line_shape, 'sample');

  const capped = await s.callTool('frame_parts', { seconds: 1000 });
  assert.equal(capped.value.seconds, 120);
});

test('live_get names every switch, its kind, its value and whether the marker acknowledged it', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const control = writeControl(
    dir,
    '# the boat run\nVICTORY_KEEP_TEXTURES=on\nVICTORY_TEXTURE_LOW=70\nVICTORY_TEXTURE_HIGH=90\n',
  );
  const s = startServer(t, {
    args: ['--log', log, '--file', control, '--lock', path.join(dir, 'run.lock')],
  });

  const { value, isError } = await s.callTool('live_get');
  assert.equal(isError, false);
  assert.deepEqual(
    value.names.map((entry) => entry.name),
    LIVE_NAMES,
  );
  const byName = new Map(value.names.map((entry) => [entry.name, entry]));
  assert.equal(byName.get('VICTORY_KEEP_TEXTURES').kind, 'on/off');
  assert.equal(byName.get('VICTORY_TEXTURE_LOW').kind, 'integer');
  assert.equal(byName.get('LIVE_SKIP_DRAW').kind, 'id list');
  assert.equal(byName.get('VICTORY_KEEP_TEXTURES').value, 'on');
  assert.equal(byName.get('VICTORY_KEEP_TEXTURES').acknowledged, true);
  assert.equal(byName.get('VICTORY_KEEP_TEXTURES').frame, 151);
  assert.equal(byName.get('VICTORY_TEXTURE_LOW').acknowledged, true, 'the segment in-force value matches');
  assert.equal(byName.get('VICTORY_TEXTURE_HIGH').acknowledged, false, '90 is requested and 80 is in force');
  assert.deepEqual(value.pending, ['VICTORY_TEXTURE_HIGH']);
  assert.equal(value.control.present, true);
});

for (const shape of ['sample', 'interval'])
test(`live_set writes through live.mjs, waits for the marker and returns old, new and frame (${shape} shape)`, async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const control = writeControl(dir, '# the boat run\nVICTORY_KEEP_TEXTURES=on\nVICTORY_TEXTURE_LOW=70\n');
  const fake = fakeRuntime(t, { log, control, shape });
  const s = startServer(t, {
    clock: fake.clock,
    args: ['--log', log, '--file', control, '--lock', path.join(dir, 'run.lock')],
    env: { HARNESS_MCP_MARKER_TIMEOUT_MS: '15000' },
  });

  const { value, isError } = await s.callTool('live_set', { set: { VICTORY_KEEP_TEXTURES: false } });
  assert.equal(isError, false, JSON.stringify(value));
  assert.equal(value.changed, true);
  assert.equal(value.acknowledged, true);
  assert.equal(value.applied[0].name, 'VICTORY_KEEP_TEXTURES');
  assert.equal(value.applied[0].old, 'on');
  assert.equal(value.applied[0].new, 'off');
  assert.equal(value.applied[0].frame, 221);
  assert.equal(
    fs.readFileSync(control, 'utf8'),
    '# the boat run\nVICTORY_KEEP_TEXTURES=off\nVICTORY_TEXTURE_LOW=70\n',
    'comments and other names stay',
  );
});

test('live_set refuses a name outside the registry and a value the file format would cut', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const control = writeControl(dir, 'VICTORY_KEEP_TEXTURES=on\n');
  const s = startServer(t, {
    args: ['--log', log, '--file', control, '--lock', path.join(dir, 'run.lock')],
    env: { HARNESS_MCP_MARKER_TIMEOUT_MS: '300' },
  });

  const unknown = await s.callTool('live_set', { set: { NOT_A_SWITCH: '1' } });
  assert.equal(unknown.isError, true);
  assert.match(unknown.value.error, /not a live switch/);
  const badValue = await s.callTool('live_set', { set: { VICTORY_KEEP_TEXTURES: 'on#off' } });
  assert.equal(badValue.isError, true);
  assert.match(badValue.value.error, /starts a comment/);
  assert.equal(fs.readFileSync(control, 'utf8'), 'VICTORY_KEEP_TEXTURES=on\n');
});

test('live_set with nothing answering is an error and restores the control file exactly', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const original = '# keep this comment\nVICTORY_KEEP_TEXTURES=on\nVICTORY_TEXTURE_LOW=70  # low\n';
  const control = writeControl(dir, original);
  const s = startServer(t, {
    args: ['--log', log, '--file', control, '--lock', path.join(dir, 'run.lock')],
    env: { HARNESS_MCP_MARKER_TIMEOUT_MS: '300' },
  });

  const { value, isError } = await s.callTool('live_set', { set: { VICTORY_KEEP_TEXTURES: 'off' } });
  assert.equal(isError, true);
  assert.match(value.error, /not reading the control file/);
  assert.equal(value.restored, true);
  assert.equal(fs.readFileSync(control, 'utf8'), original);
  assert.deepEqual(
    fs.readdirSync(dir).filter((name) => name.includes('.tmp-')),
    [],
    'no temporary file is left behind',
  );
});

for (const shape of ['sample', 'interval'])
test(`run_plan runs with the driver stop rules and returns its verdict rows (${shape} shape)`, async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const control = writeControl(dir, '');
  const fake = fakeRuntime(t, { log, control, shape });
  const s = startServer(t, {
    clock: fake.clock,
    args: ['--log', log, '--file', control, '--lock', path.join(dir, 'run.lock')],
    env: {
      HARNESS_MCP_HOLD_SCALE: '0.01',
      HARNESS_MCP_MARKER_TIMEOUT_MS: '1000',
      HARNESS_MCP_POLL_MS: '20',
      HARNESS_MCP_STALL_MS: '3000',
    },
  });

  const plan = {
    baseline: { VICTORY_KEEP_TEXTURES: 'on' },
    steps: [{ label: 'skip draws', set: { VICTORY_KEEP_TEXTURES: 'off' }, hold_s: 10 }],
  };
  const { value, isError } = await s.callTool('run_plan', { plan });
  assert.equal(isError, false, JSON.stringify(value));
  assert.equal(value.restored, true);
  assert.match(value.restore_note, /confirmed by the stats file/);
  assert.equal(value.rows.length, 1);
  assert.equal(value.rows[0].label, 'skip draws');
  // Corrected (was the old four words): the report's verdicts are these five.
  assert.ok(
    ['faster', 'slower', 'no clear change', 'too few samples', 'unreliable'].includes(value.rows[0].verdict),
    value.rows[0].verdict,
  );
  assert.equal(fs.existsSync(`${log}.live-drive.json`), true, 'the plan sidecar is beside the log');
});

test('run_plan and live_set are refused while a plan is running', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const control = writeControl(dir, '');
  const fake = fakeRuntime(t, { log, control });
  const s = startServer(t, {
    clock: fake.clock,
    args: ['--log', log, '--file', control, '--lock', path.join(dir, 'run.lock')],
    env: {
      HARNESS_MCP_HOLD_SCALE: '0.05',
      HARNESS_MCP_MARKER_TIMEOUT_MS: '2000',
      HARNESS_MCP_POLL_MS: '20',
      HARNESS_MCP_STALL_MS: '5000',
    },
  });
  t.after(() => fake.stop());

  const plan = {
    baseline: { VICTORY_KEEP_TEXTURES: 'on' },
    steps: [{ label: 'long enough', set: { VICTORY_KEEP_TEXTURES: 'off' }, hold_s: 10 }],
  };
  const running = s.callTool('run_plan', { plan });
  await fake.clock.sleep(120);
  const set = await s.callTool('live_set', { set: { VICTORY_KEEP_TEXTURES: 'on' } });
  assert.equal(set.isError, true);
  assert.match(set.value.error, /plan is running/);
  const again = await s.callTool('run_plan', { plan });
  assert.equal(again.isError, true);
  assert.match(again.value.error, /plan is already running/);
  const finished = await running;
  assert.equal(finished.isError, false, JSON.stringify(finished.value));
});

test('run_plan refuses a plan over the driver limits and an unknown switch', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const control = writeControl(dir, '');
  const s = startServer(t, {
    args: ['--log', log, '--file', control, '--lock', path.join(dir, 'run.lock')],
  });

  const step = (label) => ({ label, set: { VICTORY_TEXTURE_LOW: '60' }, hold_s: 30 });
  const many = await s.callTool('run_plan', {
    plan: { steps: Array.from({ length: 41 }, (_, index) => step(`s${index}`)) },
  });
  assert.equal(many.isError, true);
  assert.match(many.value.error, /limit is 40/);
  const unknown = await s.callTool('run_plan', {
    plan: { steps: [{ label: 'x', set: { NOT_A_SWITCH: '1' }, hold_s: 30 }] },
  });
  assert.equal(unknown.isError, true);
  assert.match(unknown.value.error, /not a live switch/);
  const noStats = await s.callTool('run_plan', { plan: { steps: [step('fine')] } });
  assert.equal(noStats.isError, true);
  assert.match(noStats.value.error, /no stats file/);
  assert.equal(fs.readFileSync(control, 'utf8'), '', 'a refused plan writes nothing');
});

test('send_keys reports LIVE_KEYS as unavailable in a build whose registry lacks it', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const control = writeControl(dir, 'VICTORY_KEEP_TEXTURES=on\n');
  const s = startServer(t, {
    args: ['--log', log, '--file', control, '--lock', path.join(dir, 'run.lock')],
    env: { HARNESS_MCP_MARKER_TIMEOUT_MS: '500' },
  });

  const { value, isError } = await s.callTool('send_keys', { sequence: '0:W,2:W' });
  assert.equal(isError, true);
  if (LIVE_NAMES.includes('LIVE_KEYS')) {
    assert.doesNotMatch(value.error, /not available in this build/);
  } else {
    assert.match(value.error, /not available in this build/);
    assert.equal(fs.readFileSync(control, 'utf8'), 'VICTORY_KEEP_TEXTURES=on\n', 'the control file is untouched');
  }
});

test('segments returns live-report objects newest last, with short stretches marked', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const text = fs.readFileSync(log, 'utf8');
  fs.writeFileSync(
    log,
    `${text}${segmentLine(2, 152, 252, 15.0, {
      VICTORY_KEEP_TEXTURES: 'off',
      VICTORY_TEXTURE_LOW: '70',
      VICTORY_TEXTURE_HIGH: '80',
      LIVE_SKIP_DISPATCH: '',
      LIVE_SKIP_DRAW: '',
    })}\n`,
  );
  const s = startServer(t, { args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });

  const { value, isError } = await s.callTool('segments');
  assert.equal(isError, false);
  assert.equal(Array.isArray(value), true);
  assert.equal(value.length, 2);
  assert.equal(value[0].segment, 1);
  assert.equal(value[0].short, false);
  assert.equal(value[1].segment, 2);
  assert.equal(value[1].short, true, '101 frames is under the 120-frame short mark');
  assert.equal(value[1].meanChange, -1.7);
  assert.equal(value[1].changes.length, 1, 'the marker between the segments belongs to the second');
});

// ----------------------------------------------------------------------------------------------------------------
// The known input: every tool's answer, field by field, against knownIntervalObject and knownLog.

test('known input, segments: the counted lines and the open one under way, last, field by field', async (t) => {
  const dir = tempDir(t);
  const log = knownLog(dir);
  const s = startServer(t, { args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });

  const { value, isError } = await s.callTool('segments');
  assert.equal(isError, false);
  assert.deepEqual(value.map((segment) => [segment.segment, segment.open]), [[5, false], [6, false], [7, true]]);
  const [five, six, seven] = value;
  assert.deepEqual(
    {
      shape: five.shape, frames: five.frames, counted: five.counted, transitionExcluded: five.transitionExcluded,
      transitionMean: five.transitionMean, lightExcluded: five.lightExcluded, lightMean: five.lightMean,
      longExcluded: five.longExcluded, longMean: five.longMean, mean: five.mean, p50: five.p50, p95: five.p95,
      begun: five.begun, ended: five.ended, short: five.short, meanChange: five.meanChange, changes: five.changes,
    },
    {
      shape: 'counted', frames: '4000..4499', counted: 497, transitionExcluded: 3, transitionMean: 88.5, lightExcluded: 2,
      lightMean: 120, longExcluded: 0, longMean: null, mean: 36.2, p50: 35.9, p95: 40.4, begun: 'change', ended: 'change',
      short: false, meanChange: null, changes: [],
    },
  );
  assert.equal(five.inForce.VICTORY_TEXTURE_LOW, '70');
  assert.equal(six.meanChange, -0.4);
  assert.deepEqual(six.changes, [{ name: 'VICTORY_TEXTURE_LOW', old: '70', new: '64', frame: 4500 }]);
  assert.equal(seven.counted, 52);
  assert.equal(seven.mean, 35.5);
  assert.equal(seven.p95, 39.2);
  assert.equal(seven.ended, null, 'not ended: the segment under way');
  assert.equal(seven.short, true, '52 counted is under the 120-frame mark');
  assert.equal(seven.meanChange, -0.3);
  assert.deepEqual(
    seven.changes.map((change) => `${change.name} ${change.old}->${change.new} @${change.frame}`),
    ['VICTORY_KEEP_TEXTURES on->off @4950', 'VICTORY_TEXTURE_LOW 64->61 @4950'],
  );
  assert.equal(seven.inForce.VICTORY_KEEP_TEXTURES, 'off');
});

test('known input, game_status from the log alone: the new frame line, its budget line and the open segment', async (t) => {
  const dir = tempDir(t);
  const log = knownLog(dir);
  const s = startServer(t, { args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });

  const { value } = await s.callTool('game_status');
  assert.equal(value.source, 'log');
  assert.equal(value.stats_shape, null, 'no stats file');
  assert.equal(value.frame_line_shape, 'interval');
  assert.equal(value.fps, 28.57);
  assert.equal(value.frame_ms, 35);
  assert.equal(value.frame_ms_p50, 34.1);
  assert.equal(value.frame_ms_p95, 38.7);
  assert.equal(value.frame, 5004);
  assert.deepEqual(value.interval, { first_frame: 5001, last_frame: 5004, frames: 4, wall_ms: 140 });
  assert.equal(value.draws, 6123);
  assert.deepEqual(value.summed_across_threads, {
    preparing_ms: 2.25, copies_ms: 1.75, copies_mb: 3.5, textures_ms: 0.5, textures_mb: 0.25, host_waits_ms: 6.5,
    copy_back_ms: 0.75, copy_back_mb: 1.25,
  });
  // The runtime's budget line: the title and its calls have no processor time of their own; a phase the line's
  // processor list leaves out had none.
  assert.deepEqual(value.parts, {
    title: { ms: 21.25, cpu_ms: null }, list_calls: { ms: 3.13, cpu_ms: null }, unaccounted: { ms: 1.37, cpu_ms: 0 },
    relay_wait: { ms: 9.25, cpu_ms: 0 },
  });
  assert.equal(value.parts_check.holds, true);
  // Corrected (was undefined: the open line was pushed as a closed segment with no number; then `segment`).
  assert.equal(value.segments_closed_total, 6, 'the last closed segment');
  assert.equal(value.segment_open.segment, 7);
  assert.equal(value.segment_open.counted, 52);
  assert.equal(value.segment_open.mean_ms, 35.5);
  assert.equal(value.segment_open.transition_mean_ms, 61.5);
  assert.equal(value.segment_open.light_excluded, 1);
  assert.equal(value.segment_open.light_mean_ms, 120);
  assert.equal('frame_ms_worst' in value, false, 'a frame line has a p95 of its window, not a worst frame');
  assert.equal(value.segment_open.source, 'log');
  assert.equal(value.live.VICTORY_KEEP_TEXTURES, 'off', 'in force now: the open line, newer than the closed one');
  assert.equal(value.dispatched, null, 'the frame line carries no per-frame dispatch count');
});

test('known input, game_status from the stats object: every field it reads, units in the names', async (t) => {
  const dir = tempDir(t);
  const log = knownLog(dir);
  const clock = virtualClock();
  const stats = cannedStats(dir, knownIntervalObject(clock.now()));
  const s = startServer(t, { clock, args: ['--log', log, '--stats', stats, '--lock', path.join(dir, 'run.lock')] });

  const { value } = await s.callTool('game_status');
  assert.equal(value.source, 'stats');
  assert.equal(value.stats_shape, 'interval');
  assert.equal(value.seq, 731);
  assert.equal(value.stats_age_by, 'unix_ms');
  assert.equal(value.stats_age_s, 2, 'stats_age_s');
  assert.equal(value.stats_stale, false);
  assert.equal(value.fps, 28.57);
  assert.equal(value.frame_ms, 35);
  assert.equal(value.frame_ms_p50, 34.1);
  assert.equal(value.frame_ms_worst, 38.7);
  assert.equal('frame_ms_p95' in value, false);
  assert.equal(value.frame, 5004);
  assert.deepEqual(value.interval, { first_frame: 5001, last_frame: 5004, frames: 4, wall_ms: 140 });
  assert.equal(value.draws, 6123);
  assert.equal(value.dispatched, 287);
  assert.equal(value.dispatches_since_previous, null);
  assert.equal(value.vram_mb, 17777);
  assert.deepEqual(value.parts.title, { ms: 21.25, cpu_ms: null });
  assert.deepEqual(value.parts.list_calls, { ms: 3.125, cpu_ms: null });
  assert.deepEqual(value.parts.unaccounted, { ms: 1.375, cpu_ms: 0.5 });
  assert.deepEqual(value.parts.relay_wait, { ms: 9.25, cpu_ms: 0.75 });
  assert.deepEqual(Object.keys(value.parts), RUNTIME_PHASES.filter((name) => name !== 'pool_work' && name !== 'relay_land'), 'every phase, in the runtime\'s order');
  assert.deepEqual(value.parts_check, { frame_ms: 35, phases_sum_ms: 35, holds: true });
  assert.deepEqual(value.summed_across_threads, {
    preparing_ms: 2.25, copies_ms: 1.75, host_wait_ms: 6.5, fetch_queue_fallback_by: { numbered: 0.5 },
  });
  assert.deepEqual(value.live, knownIntervalObject().live);
  assert.equal(value.segments_closed_total, 6);
  assert.deepEqual(value.segment_open, {
    segment: 7, first_frame: 4950, last_frame: 5004, counted: 52, transition_excluded: 3, transition_mean_ms: 61.5,
    light_excluded: 1, light_mean_ms: 120, long_excluded: 0, long_mean_ms: null, mean_ms: 35.5, p50_ms: 34.6,
    p95_ms: 39.2, begun: 'change', source: 'stats',
  });
  assert.deepEqual(value.absent_optional, ['hash_check', 'texture_receipts', 'landed_classes', 'texture_recheck', 'textures_made']);
  assert.match(value.absent_optional_note, /switch is off/);
  assert.deepEqual(value.kernel_calls, { per_frame: { reads: 1.5 } });
  assert.equal(value.stats.stores_failed_total, 3, 'the whole object passes through');
});

test('known input, frame_parts from the stats object and from the log alone', async (t) => {
  const dir = tempDir(t);
  const log = knownLog(dir);
  const stats = cannedStats(dir, knownIntervalObject());
  const s = startServer(t, { args: ['--log', log, '--stats', stats, '--lock', path.join(dir, 'run.lock')] });

  const { value } = await s.callTool('frame_parts', { seconds: 10 });
  assert.equal(value.source, 'stats');
  assert.equal(value.stats_shape, 'interval');
  assert.equal(value.samples, 1);
  assert.equal(value.frames, 4);
  assert.equal(value.wall_ms, 140);
  assert.equal(value.objects_missed, 0);
  assert.equal(value.stats_stale, false);
  assert.deepEqual(value.frame, { first: 5001, last: 5004 });
  assert.deepEqual(value.frame_ms, { mean: 35, median: 34.1, p95: 38.7, samples: 1, frames: 4, list_frames: 4, list_missing: 0 });
  assert.deepEqual(value.parts.title, { mean: 21.25, cpu_ms: null, median: 21.25, p95: 21.25, samples: 1 });
  assert.equal(value.parts.list_calls.cpu_ms, null);
  assert.equal(value.parts.relay_wait.mean, 9.25);
  assert.equal(value.budget.identity, 'holds');
  assert.equal(value.budget.frame_ms, 35);
  assert.equal(value.budget.phases_sum_ms, 35);
  assert.equal(value.budget.objects, 1);
  assert.match(value.parts_note, /they are the parts of the frame/);
  assert.equal(value.summed_across_threads.preparing_ms.mean, 2.25);
  assert.equal(value.summed_across_threads['fetch_queue_fallback_by.numbered'].mean, 0.5);
  assert.match(value.summed_note, /not parts of it/);

  // The log alone: a folder with no stats file beside the log (the server would find one there).
  const bare = tempDir(t);
  const logOnly = startServer(t, { args: ['--log', knownLog(bare), '--lock', path.join(bare, 'run.lock')] });
  const fromLog = (await logOnly.callTool('frame_parts', { seconds: 10 })).value;
  assert.equal(fromLog.source, 'log');
  assert.equal(fromLog.frame_line_shape, 'interval');
  assert.equal(fromLog.samples, 2);
  assert.equal(fromLog.frames, 40);
  assert.equal(fromLog.frame_ms.mean, 35.9, '(36 x 36.0 + 4 x 35.0) / 40 frames');
  assert.equal(fromLog.frame_ms.median, null, 'two lines\' own percentiles do not make an arm\'s');
  assert.match(fromLog.frame_ms_note, /frame_ms_list/);
  assert.equal(fromLog.parts.title.mean, 21.925, '(22 x 36 + 21.25 x 4) / 40');
  assert.equal(fromLog.budget.identity, 'holds');
  assert.equal(fromLog.summed_across_threads.preparing_ms.mean, 2.25);
});

test('known input, live_get: values from the stats object, acknowledgements from the markers, sources named', async (t) => {
  const dir = tempDir(t);
  const log = knownLog(dir);
  const stats = cannedStats(dir, knownIntervalObject());
  const control = writeControl(dir, 'VICTORY_KEEP_TEXTURES=off\nVICTORY_TEXTURE_LOW=61\nVICTORY_TEXTURE_HIGH=88\n');
  const s = startServer(t, { args: ['--log', log, '--stats', stats, '--file', control, '--lock', path.join(dir, 'run.lock')] });

  const { value } = await s.callTool('live_get');
  const byName = new Map(value.names.map((entry) => [entry.name, entry]));
  assert.deepEqual(byName.get('VICTORY_KEEP_TEXTURES'), {
    name: 'VICTORY_KEEP_TEXTURES', kind: 'on/off', value: 'off', requested: 'off', source: 'log', acknowledged: true, frame: 4950,
  });
  assert.equal(byName.get('VICTORY_TEXTURE_LOW').frame, 4950);
  // Corrected (was source 'log'): a value read from the stats object's live map is named as from the stats file.
  assert.deepEqual(byName.get('VICTORY_TEXTURE_HIGH'), {
    name: 'VICTORY_TEXTURE_HIGH', kind: 'integer', value: '85', requested: '88', source: 'stats', acknowledged: false, frame: null,
  });
  assert.deepEqual(byName.get('VICTORY_LATE_WAITS'), {
    name: 'VICTORY_LATE_WAITS', kind: 'on/off', value: 'on', requested: null, source: 'stats', acknowledged: true, frame: null,
  });
  assert.deepEqual(value.pending, ['VICTORY_TEXTURE_HIGH']);
  assert.deepEqual(value.control, { present: true, assignments: 3 });
  assert.equal(byName.get('LIVE_MARK').kind, 'mark', 'the mark is a live name, of its own kind');
  assert.equal(value.stats_stale, false);
});

test('live_get reads a requested value as the runtime holds it, and marks the values of a stale object as last known', async (t) => {
  const dir = tempDir(t);
  const log = knownLog(dir);
  const stats = cannedStats(dir, { ...knownIntervalObject(), live: { VICTORY_FETCH_QUEUE: 'on', VICTORY_LAND_SCAN: 'off' } });
  const control = writeControl(dir, 'VICTORY_FETCH_QUEUE=1\n');
  const s = startServer(t, { args: ['--log', log, '--stats', stats, '--file', control, '--lock', path.join(dir, 'run.lock')] });
  const fresh = (await s.callTool('live_get')).value;
  const queue = fresh.names.find((entry) => entry.name === 'VICTORY_FETCH_QUEUE');
  assert.equal(queue.acknowledged, true, 'the runtime reads 1 as on');
  assert.deepEqual(fresh.pending, []);
  // Ten minutes old by its own clock: shown, as the last known, and acknowledging nothing.
  cannedStats(dir, { ...knownIntervalObject(Date.now() - 600_000), seq: 900, live: { VICTORY_FETCH_QUEUE: 'on', VICTORY_LAND_SCAN: 'off' } });
  const old = (await s.callTool('live_get')).value;
  const scan = old.names.find((entry) => entry.name === 'VICTORY_LAND_SCAN');
  assert.deepEqual([scan.value, scan.source], ['off', 'stats']);
  assert.equal(old.stats_stale, true);
  assert.match(old.stale_note, /last known, not confirmed in force now/);
  assert.deepEqual(old.pending, ['VICTORY_FETCH_QUEUE'], 'a stale object confirms no request');
});

test('known input, log_find: the known warning, masked', async (t) => {
  const dir = tempDir(t);
  const log = knownLog(dir);
  const s = startServer(t, { child: true, args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });
  const { value } = await s.callTool('log_find', { pattern: 'a known warning' });
  assert.equal(value.total, 1);
  assert.deepEqual(value.lines, [`${FRAME_PREFIX}08Z  WARN victory_gpu::host: a known warning for log_find, shader <id> skipped`]);
});

test('known input, send_keys: the marker frame, the done frame and the presses played (interval shape)', async (t) => {
  if (!LIVE_NAMES.includes('LIVE_KEYS')) return;
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const control = writeControl(dir, 'VICTORY_KEEP_TEXTURES=on\n');
  const fake = fakeRuntime(t, { log, control, shape: 'interval' });
  const s = startServer(t, {
    clock: fake.clock,
    args: ['--log', log, '--file', control, '--lock', path.join(dir, 'run.lock')],
    env: { HARNESS_MCP_MARKER_TIMEOUT_MS: '15000' },
  });
  const { value, isError } = await s.callTool('send_keys', { sequence: '0:W,1:A:0.5' });
  assert.equal(isError, false, JSON.stringify(value));
  assert.equal(value.changed, true);
  assert.equal(value.presses, 2);
  assert.equal(value.length_s, 1.5);
  assert.equal(value.frame, 221, 'the LIVE_KEYS marker\'s frame');
  assert.equal(value.done_frame, 251);
  assert.equal(value.presses_played, 2);
});

test('log_find masks a drive path and a 16-hex id, and returns at most 40 lines', async (t) => {
  const dir = tempDir(t);
  const privatePath = ['Z:', 'made', 'up', 'private-run.log'].join('/');
  const extra = [
    `WARN opened ${privatePath} for shader deadbeefcafebabe and stopped`,
    // Made-up: a folder name with a space, unquoted; then addresses on reserved names, and a line without one.
    `WARN unquoted ${['Z:', 'Private Stuff', 'dump', 'other.bin'].join('\\')} then 3 ms`,
    'WARN fetched https://example.com/a' + Q + 'b=1 and gave up',
    'WARN fetched HTTP://EXAMPLE.ORG/x',
    'WARN fetched nothing',
    "priority: above normal for pid 4242 (this run's game)",
    ...Array.from({ length: 50 }, (_, index) => `WARN counter ${index}`),
    'INFO all good',
  ];
  const log = cannedLog(dir, { extra });
  const s = startServer(t, { args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });

  const privateHit = await s.callTool('log_find', { pattern: 'deadbeefcafebabe' });
  assert.equal(privateHit.isError, false);
  assert.equal(privateHit.value.total, 1);
  const line = privateHit.value.lines[0];
  assert.match(line, /<path>/);
  assert.match(line, /<id>/);
  assert.equal(line.includes(privatePath), false);
  assert.equal(line.includes('deadbeefcafebabe'), false);

  const spaced = await s.callTool('log_find', { pattern: 'unquoted' });
  assert.equal(spaced.value.lines[0], 'WARN unquoted <path> then 3 ms', 'a folder name with a space leaks no tail');

  const addressed = await s.callTool('log_find', { pattern: 'fetched' });
  assert.equal(addressed.value.total, 1, 'the line without an address');
  assert.equal(addressed.value.withheld, 2);
  assert.match(addressed.value.withheld_note, /2 lines withheld/);
  assert.equal(JSON.stringify(addressed.value).includes('example'), false);

  const priority = await s.callTool('log_find', { pattern: 'priority' });
  assert.equal(priority.value.lines[0], "priority: above normal for pid <pid> (this run's game)", 'no process id, as game_status gives none');

  const counters = await s.callTool('log_find', { pattern: 'counter', last: 1000 });
  assert.equal(counters.value.total, 50);
  assert.equal(counters.value.lines.length, 40, 'at most 40 lines come back');
  assert.match(counters.value.lines.at(-1), /counter 49/, 'newest match is last');

  const pattern = await s.callTool('log_find', { pattern: '/counter \\d+/' });
  assert.equal(pattern.value.total, 50);
  const unsafe = await s.callTool('log_find', { pattern: '/(a+)+$/' });
  assert.equal(unsafe.isError, true);
  assert.match(unsafe.value.error, /safe pattern/);
});

test('malformed JSON and an unknown method do not crash the server, and stdout stays protocol-only', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const s = startServer(t, { child: true, args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });

  s.child.stdin.write('{not json\n');
  const unknown = await s.request('does/not/exist');
  assert.equal(unknown.error.code, -32601);
  const ping = await s.request('ping');
  assert.deepEqual(ping.result, {});
  const parseError = s.lines.find((message) => message.error && message.error.code === -32700);
  assert.ok(parseError, 'the malformed line got a parse error');
  assert.equal(parseError.id, null);
  assert.deepEqual(s.nonJson, [], 'nothing but JSON-RPC lines appeared on stdout');
  for (const message of s.lines) assert.equal(message.jsonrpc, '2.0');
});

test('an unknown tool is a tool error, not a dropped call', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const s = startServer(t, { args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });

  const { value, isError } = await s.callTool('not_a_tool');
  assert.equal(isError, true);
  assert.match(value.error, /unknown tool/);
});

test('the server exits when the client closes its input', async (t) => {
  const dir = tempDir(t);
  const log = cannedLog(dir);
  const s = startServer(t, { child: true, args: ['--log', log, '--lock', path.join(dir, 'run.lock')] });
  await s.request('ping');
  await s.stop();
  assert.notEqual(s.child.exitCode, null);
  assert.deepEqual(s.nonJson, []);
});

test('run-world --live gives the runtime the stats file beside the control file', (t) => {
  const dir = tempDir(t);
  const bash = spawnSync('bash', ['-c', 'command -v bash'], { encoding: 'utf8' }).stdout.trim();
  fs.writeFileSync(path.join(dir, 'victory.toml'), '');
  const result = spawnSync('bash', [path.join(repo, 'harness', 'run-world.sh'), '--live', '--print-env'], {
    encoding: 'utf8',
    env: isolatedEnv({
      VICTORY_BIN: bash,
      VICTORY_DIR: dir,
      VICTORY_CONFIG: path.join(dir, 'victory.toml'),
      GAME_EXE: path.join(dir, 'game.exe'),
      VICTORY_CAPTURES: path.join(dir, 'captures'),
    }),
  });
  assert.equal(result.status, 0, result.stderr);
  const lines = result.stdout.split(/\r?\n/);
  const control = lines.find((line) => line.startsWith('VICTORY_LIVE='));
  const stats = lines.find((line) => line.startsWith('VICTORY_LIVE_STATS='));
  assert.ok(control, 'the runtime receives VICTORY_LIVE');
  assert.ok(stats, 'the runtime receives VICTORY_LIVE_STATS');
  assert.match(control, /\/live\.txt$/);
  assert.match(stats, /\/stats\.json$/);
  assert.match(result.stdout, /^live:\s+live\.txt$/m);
  const folders = fs.readdirSync(path.join(dir, 'captures'));
  assert.equal(folders.length, 1);
  assert.equal(fs.existsSync(path.join(dir, 'captures', folders[0], 'live.txt')), true);
  assert.equal(fs.existsSync(path.join(dir, 'captures', folders[0], 'stats.json')), false, 'the runtime writes it');
});


// The cost part's fourth figure is the host's waits (the audit of 2026-11-26), under either label the runtime wrote.
test('the frame reader names the host waits host_waits_ms, and no part calls it GPU time', async () => {
  const { scanFrames } = await import('./live-server.mjs');
  const waits = "7.5 host waits (for the relay 1.5, for flights' fence 2.0, synchronous submissions sent to retired 4.0)";
  const [old, current] = scanFrames(
    [frameLine(0, 100, 60.0, 16.7), frameLine(5, 160, 60.0, 16.7).replace('0.9 on the GPU', waits)].join('\n'),
  );
  // Corrected (was frame.parts): the frame line's figures are summed across threads, not parts of the frame.
  assert.equal(old.summed.host_waits_ms, 0.9);
  assert.equal(current.summed.host_waits_ms, 7.5);
  for (const frame of [old, current]) assert.deepEqual(Object.keys(frame.summed).filter((key) => /gpu/i.test(key)), []);
  assert.equal('parts' in current, false, 'the old name is gone');
});
