#!/usr/bin/env node
// A Model Context Protocol server over the harness's live tools: numbers out, switches in.
//
//   node harness/mcp/live-server.mjs [--log <run log>] [--file <control file>] [--stats <stats file>]
//                                    [--lock <run lock>]
//
// It speaks newline-delimited JSON-RPC 2.0 on standard input and output and has no dependencies beyond Node.
// Standard output carries protocol messages only; diagnostics go to standard error. The run is found from
// harness/run.lock and the newest harness/logs/*.log; the control file is the live.txt beside that log's capture
// folder and the stats file is stats.json (or live-stats.json) there. --log, --file and --stats (or HARNESS_LOG,
// VICTORY_LIVE and VICTORY_LIVE_STATS) override the search for a test or a caller that knows the run.
//
// There is deliberately no tool that starts, ends or kills a run, and no tool that takes a file path: the only
// file a tool writes is the live control file (and live-drive's sidecar beside the run log during a plan).
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

import { loadLocalEnv } from '../local-env.mjs';
import { checkName, checkValue, mergeControl, parseAssignment, readControl, writeAtomic } from '../live.mjs';
import { normalizePlan, reportRows, runPlan, loadTimeline } from '../live-drive.mjs';
import { LIVE_KINDS as KINDS, LIVE_NAMES, isLiveName, normalizeLiveValue } from '../live-names.mjs';
import { parseLiveLine, parseLiveLog } from '../live-report.mjs';
import { COST_RE, FRAME_RE, median, parseBudgetLine as parseBudget, parseFrameWindow, percentile, stripAnsi } from '../parse-run.mjs';
import { OPTIONAL_PARTS, RUNTIME_PHASES, objectAgeMs, objectsFigures, readStatsObject, sequenceOf } from '../stats-object.mjs';
import { withheldText, withoutWebAddresses } from '../web-address.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const HARNESS_DIR = path.resolve(HERE, '..');
const DEFAULT_LOGS = path.join(HARNESS_DIR, 'logs');
const DEFAULT_LOCK = path.join(HARNESS_DIR, 'run.lock');
const DEFAULT_CAPTURES = 'victory-captures';

export const SERVER_INFO = Object.freeze({ name: 'victory-live', version: '1.0.0' });
export const MARKER_TIMEOUT_MS = 15_000;
export const MAX_FRAME_WINDOW_S = 120;
export const MAX_LOG_FIND = 40;
export const MAX_KEYS_PRESSES = 64;
export const MAX_KEYS_S = 120;
// A stats object older than this is stale, not current: the interval shape is written about once a second (its age
// by its own clock, unix_ms, or how long its seq has stood); the older sample about once in six (its file's time).
// HARNESS_MCP_STALE_S overrides the interval shape's limit, for a test.
export const STALE_S = Object.freeze({ interval: 5, sample: 15 });
// Rounding of a budget phase as the stats object writes it (3 decimals): the identity check's allowance per figure.
const STATS_ROUNDING_MS = 0.0005;

function staleLimits(env) {
  const override = Number(env.HARNESS_MCP_STALE_S);
  return override > 0 ? { ...STALE_S, interval: override } : STALE_S;
}

const INSTRUCTIONS =
  'Reads the live run\'s numbers and flips its switches. There is deliberately no tool that starts, ends or ' +
  'kills a run, and none that reads or writes any file other than the control file and a plan\'s sidecar. ' +
  'Private paths and ids come back redacted.';

// The kind of each live name, for live_get: live-names.mjs's table (checked against the runtime's `switches!`), in
// words. An unknown name is text.
const KIND_WORDS = Object.freeze({ onoff: 'on/off', integer: 'integer', list: 'id list', keys: 'sequence', mark: 'mark' });
const LIVE_KINDS = Object.freeze(Object.fromEntries(Object.entries(KINDS).map(([name, kind]) => [name, KIND_WORDS[kind] ?? kind])));

/// A tool-level failure: the call answers with isError true and this message, never a JSON-RPC error.
export class ToolError extends Error {
  constructor(message, detail = {}) {
    super(message);
    this.detail = detail;
  }
}

/// The real clock. A test passes its own (live-fake-runtime.mjs virtualClock: now, sleep, every) to createProtocol, so
/// that no wait or age in the tools depends on how fast the machine is.
const REAL_CLOCK = Object.freeze({
  now: () => Date.now(),
  sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  every(ms, fn) {
    const timer = setInterval(fn, ms);
    if (timer.unref) timer.unref();
    return () => clearInterval(timer);
  },
});
const round = (value, digits = 3) => {
  if (!Number.isFinite(value)) return null;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
};

// ----------------------------------------------------------------------------------------------------------------
// paths and the run in front of us

function newestLog(dir) {
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return null;
  }
  let best = null;
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.log')) continue;
    const file = path.join(dir, entry.name);
    let stat;
    try {
      stat = fs.statSync(file);
    } catch {
      continue;
    }
    if (!best || stat.mtimeMs > best.mtimeMs) best = { file, mtimeMs: stat.mtimeMs };
  }
  return best ? best.file : null;
}

let localEnvLoaded = false;
function capturesRoot(env) {
  if (env.VICTORY_CAPTURES) return env.VICTORY_CAPTURES;
  if (!localEnvLoaded) {
    localEnvLoaded = true;
    try {
      loadLocalEnv({ env, warn: () => {} });
    } catch {
      // A machine without harness/local.env keeps the generic relative default.
    }
  }
  return env.VICTORY_CAPTURES || DEFAULT_CAPTURES;
}

/// The capture folder run-world.sh made: same name as the log, without its extension.
function captureFolder(logFile, env) {
  if (!logFile) return null;
  return path.join(capturesRoot(env), path.basename(logFile).replace(/\.log$/, ''));
}

function existing(file) {
  try {
    return fs.statSync(file).isFile();
  } catch {
    return false;
  }
}

/// The file to read and write for this call. The overrides are for tests and for a caller that already knows
/// the run; otherwise the newest log and the capture folder beside it decide.
export function resolveRun(options = {}, env = process.env) {
  const logFile = options.log || env.HARNESS_LOG || newestLog(DEFAULT_LOGS);
  const capture = captureFolder(logFile, env);
  let controlFile = options.control || env.VICTORY_LIVE || null;
  if (!controlFile && capture) {
    const candidate = path.join(capture, 'live.txt');
    if (existing(candidate)) controlFile = candidate;
  }
  if (!controlFile && logFile) {
    const candidate = path.join(path.dirname(logFile), 'live.txt');
    if (existing(candidate)) controlFile = candidate;
  }
  let statsFile = options.stats || env.VICTORY_LIVE_STATS || null;
  if (!statsFile && capture) {
    for (const name of ['stats.json', 'live-stats.json']) {
      const candidate = path.join(capture, name);
      if (existing(candidate)) {
        statsFile = candidate;
        break;
      }
    }
  }
  if (!statsFile && logFile) {
    for (const name of ['stats.json', 'live-stats.json']) {
      const candidate = path.join(path.dirname(logFile), name);
      if (existing(candidate)) {
        statsFile = candidate;
        break;
      }
    }
  }
  const lockFile = options.lock || env.VICTORY_RUN_LOCK || env.HARNESS_RUN_LOCK || DEFAULT_LOCK;
  return { logFile, controlFile, statsFile, lockFile };
}

function readText(file) {
  try {
    return fs.readFileSync(file, 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    throw new ToolError(`cannot read the run file (${err.code || 'read error'})`);
  }
}

function errorCode(err) {
  return err && err.code ? String(err.code) : 'read error';
}

function requireLog(run) {
  if (!run.logFile || !existing(run.logFile)) {
    throw new ToolError('no run log: no run has started, or this server cannot see its log');
  }
}

// ----------------------------------------------------------------------------------------------------------------
// reading the log

/// Every `live:` marker and closed `live segment` line in a log, in order, and the segment still under way (`open`:
/// the last open line whose number is past the last closed segment's, with the markers since that one), or null.
/// Unlike parseLiveLog this keeps the markers that follow the last segment, which is what a freshly applied change
/// looks like. Both segment line shapes are read (live-report.mjs).
export function scanLive(text) {
  const markers = [];
  const segments = [];
  let pending = [];
  let open = null;
  let closed = 0;
  for (const raw of String(text).split(/\r?\n/)) {
    const parsed = parseLiveLine(raw);
    if (!parsed) continue;
    if (parsed.change) {
      markers.push(parsed.change);
      pending.push(parsed.change);
    } else if (parsed.openSegment) {
      // The segment under way, written after each frame line: not a closed segment.
      open = parsed.openSegment;
    } else {
      segments.push({ ...parsed.segment, changes: pending });
      closed = Math.max(closed, parsed.segment.segment);
      pending = [];
    }
  }
  return { markers, segments, open: open && open.segment > closed ? { ...open, changes: pending } : null };
}

/// A budget line (parse-run.mjs reads it, in the runtime's shape) as the stats record's budget (`stats-object.mjs`):
/// `{ frames, frameMs, sumMs, scopes, calls, largestTitleGapMs, titleAndCallsCpuMs, phases, roundingMs }`, or null.
/// The line names title and unaccounted always and another phase only when it is not zero; a phase's processor time
/// is the line's, null for the title and its calls (they have none of their own) and where the counter's rate was
/// unknown. `roundingMs` is half the last printed digit, for the identity check.
export function budgetRecordOfLine(line) {
  const b = parseBudget(line);
  if (!b) return null;
  const entry = ['title', 'list_calls', 'object_calls', 'hle_calls'];
  const phases = {};
  let decimals = Math.min(b.frameDecimals, b.sumDecimals);
  for (const phase of b.phases) {
    const cpuMs = entry.includes(phase.name) || b.cpuMissing ? null : (b.cpu[phase.name] ?? 0);
    phases[phase.name] = { ms: phase.ms, cpuMs };
    decimals = Math.min(decimals, phase.decimals);
  }
  return {
    thread: 'presenting',
    frames: b.frames,
    frameMs: b.frameMs,
    sumMs: b.sumMs,
    scopes: b.scopes,
    calls: b.calls,
    largestTitleGapMs: b.largestTitleGapMs,
    titleAndCallsCpuMs: b.titleAndCallsCpuMs,
    phases,
    roundingMs: 0.5 * 10 ** -decimals,
  };
}

/// The `harness build:`, `harness checkout:` and `harness env:` lines at the top of a run log. Values that look
/// like a path are left out (the header usually replaces them already; an older log may not), so no caller can
/// get a folder path out of game_status.
export function parseHeader(text) {
  const header = { build: null, checkout: null, profile: null, console: null, switches: {} };
  for (const raw of String(text).split(/\r?\n/)) {
    const line = stripAnsi(raw).replace(/\r$/, '');
    const build = line.match(/^harness build:\s*(.*)$/);
    if (build) {
      const rest = build[1].trim();
      const named = rest.match(/^(.*?)\s*profile=(\S*)\s+console=(\S*)(?:\s+branch=(\S*))?\s*$/);
      if (named) {
        header.build = named[1].trim() || null;
        header.profile = named[2] || null;
        header.console = named[3] || null;
      } else if (rest) {
        header.build = rest.split(/\s+/)[0];
      }
      continue;
    }
    const checkout = line.match(/^harness checkout:\s*(\S+)\s*$/);
    if (checkout) {
      header.checkout = checkout[1];
      continue;
    }
    const envLine = line.match(/^harness env:\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
    if (!envLine) continue;
    const value = envLine[2].trim();
    if (value.includes('/') || value.includes('\\') || /^[A-Za-z]:/.test(value)) continue;
    if (Object.keys(header.switches).length >= 60) continue;
    header.switches[envLine[1]] = redactLine(value);
  }
  return header;
}

function timestampMs(text) {
  const at = Date.parse(text);
  return Number.isNaN(at) ? null : at;
}

/// The frame lines, with the figures parse-run.mjs already names. A line's numbers are the only thing kept. `summed`
/// are the parenthesised figures, summed across threads (they overlap; they are not parts of the frame); `window` is
/// the clause a line from job/timing-truth on ends with (its figures are means over those frames), null for an older
/// line; `budget` is the budget line that follows it (diagnostics builds), or null.
export function scanFrames(text) {
  const frames = [];
  for (const raw of String(text).split(/\r?\n/)) {
    const line = stripAnsi(raw).replace(/\r$/, '');
    const m = line.match(FRAME_RE);
    if (!m) {
      const budget = frames.length ? budgetRecordOfLine(line) : null;
      if (budget && !frames[frames.length - 1].budget) frames[frames.length - 1].budget = budget;
      continue;
    }
    const costs = line.match(COST_RE);
    frames.push({
      ts: m[1],
      tsMs: timestampMs(m[1]),
      frame: m.groups.frame === undefined ? null : Number(m.groups.frame),
      fps: Number(m.groups.fps),
      ms: Number(m.groups.ms),
      draws: Number(m.groups.draws),
      dropped: Number(m.groups.dropped),
      window: parseFrameWindow(line),
      budget: null,
      summed: costs
        ? {
            preparing_ms: Number(costs[1]),
            copies_ms: Number(costs[2]),
            copies_mb: Number(costs[3]),
            textures_ms: Number(costs[4]),
            textures_mb: Number(costs[5]),
            host_waits_ms: Number(costs[6]),
            copy_back_ms: Number(costs[7]),
            copy_back_mb: Number(costs[8]),
          }
        : {},
    });
  }
  return frames;
}

// ----------------------------------------------------------------------------------------------------------------
// redaction (public-lines.sh's rule, as a function: no drive path, home path or long hex id leaves the server)

/// Replace path-shaped tokens and 16-or-more-digit hex ids. Used on every log_find line and on header values, so
/// a private folder or a shader/pipeline id cannot reach the client by accident.
export function redactLine(text) {
  let out = stripAnsi(String(text)).replace(/\r$/, '');
  // Quoted paths first, so a folder with a space in its name goes whole.
  out = out.replace(/"([^"\r\n]*[\\/][^"\r\n]*)"/g, '"<path>"');
  out = out.replace(/'([^'\r\n]*[\\/][^'\r\n]*)'/g, "'<path>'");
  // Windows drive paths and UNC shares.
  out = out.replace(/(?:[A-Za-z]:[\\/]|\\\\)[^\s"'`,;)\]]*/g, '<path>');
  // POSIX absolute paths, after start, whitespace, a quote or an opening bracket; not the // of a URL.
  out = out.replace(/(^|[\s"'`(=[])\/(?!\/)[^\s"'`,;)\]]*/g, '$1<path>');
  // Home-relative paths.
  out = out.replace(/(^|[\s"'`(=[])~[\\/][^\s"'`,;)\]]*/g, '$1<path>');
  // What a space split from a path (a folder's name may hold one: the check of 2026-11-26 found the tail of
  // `E:\Private Stuff\dump\x.bin` returned): the words after it that still hold a slash or backslash are its tail.
  out = out.replace(/<path>(?: +[^\s"'`,;)\]]*[\\/][^\s"'`,;)\]]*)+/g, '<path>');
  // A process id, as the harness's priority, volume and recording lines name the game's (game_status gives none).
  out = out.replace(/\b(pid|process id|processid)([ =:#]*)\d+/gi, '$1$2<pid>');
  // A 16-hex (or longer) id, the runtime's pipeline/texture hash shape.
  out = out.replace(/(?<![0-9a-fA-F])[0-9a-fA-F]{16,}(?![0-9a-fA-F])/g, '<id>');
  return out;
}

// A stats key that names the process or a file is never passed through, even though the rest of the object is
// open-ended: game_status promises no process id, command line or path.
const UNSAFE_STATS_KEY_RE =
  /^(?:pid|ppid|process|process_id|cmd|command|command_line|args|argv|exe|executable|path|filepath|file|owner|log|control|stats)$/i;

/// A value from the open-ended stats object with every string redacted, so a field this version does not know
/// still passes through without carrying a private path to the client.
export function safeValue(value) {
  if (typeof value === 'string') return redactLine(value);
  if (Array.isArray(value)) return value.map(safeValue);
  if (value && typeof value === 'object') {
    const out = {};
    for (const [key, child] of Object.entries(value)) {
      if (UNSAFE_STATS_KEY_RE.test(key)) continue;
      out[key] = safeValue(child);
    }
    return out;
  }
  return value;
}

// The stats object's optional parts (stats.rs `optional`, stats-object.mjs OPTIONAL_PARTS): present only while their
// switch is on.
const ABSENT_OPTIONAL_NOTE =
  'absent_optional: parts absent from the stats object because their switch is off (or this build has none); ' +
  'they are not zeros';

/// The segment under way, from the stats object's `segment_open` (stats-object.mjs record) or the log's open line
/// (live-report.mjs), in the stats file's names (live.rs `segment_json`). A segment with nothing counted has no
/// figures, and a kind with no frame left out no mean (null, never 0.0).
export function segmentOpenOut(open, source) {
  if (!open) return null;
  const counted = open.counted ?? null;
  const figure = (value) => (counted ? (value ?? null) : null);
  const left = (count, mean) => (count ? (mean ?? null) : null);
  return {
    segment: open.segment ?? null,
    first_frame: open.firstFrame ?? null,
    last_frame: open.lastFrame ?? null,
    counted,
    transition_excluded: open.transitionExcluded ?? null,
    transition_mean_ms: left(open.transitionExcluded, open.transitionMeanMs ?? open.transitionMean),
    light_excluded: open.lightExcluded ?? null,
    light_mean_ms: left(open.lightExcluded, open.lightMeanMs ?? open.lightMean),
    long_excluded: open.longExcluded ?? null,
    long_mean_ms: left(open.longExcluded, open.longMeanMs ?? open.longMean),
    mean_ms: figure(open.meanMs ?? open.mean),
    p50_ms: figure(open.p50Ms ?? open.p50),
    p95_ms: figure(open.p95Ms ?? open.p95),
    begun: open.begun ?? null,
    source,
  };
}

// ----------------------------------------------------------------------------------------------------------------
// statistics

function flattenNumbers(value, prefix = '', out = {}) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    out[prefix || 'value'] = value;
  } else if (value && typeof value === 'object' && !Array.isArray(value)) {
    for (const [key, child] of Object.entries(value)) {
      flattenNumbers(child, prefix ? `${prefix}.${key}` : key, out);
    }
  }
  return out;
}

function describeValues(values) {
  if (!values.length) return null;
  return {
    mean: round(values.reduce((sum, value) => sum + value, 0) / values.length),
    median: round(median(values)),
    p95: round(percentile(values, 0.95)),
    samples: values.length,
  };
}

const weightedMean = (rows) => {
  const weight = rows.reduce((sum, row) => sum + row.w, 0);
  return weight > 0 ? rows.reduce((sum, row) => sum + row.v * row.w, 0) / weight : null;
};

// What the two kinds of figure are. A reader is told on every answer that carries them.
export const SUMMED_NOTE =
  'summed_across_threads: each figure is summed across threads (a mean per frame in the interval shape); they ' +
  'overlap each other and the frame, and are not parts of it';
const PARTS_ABSENT_NOTE =
  'parts: n/a, no budget here (a build without diagnostics, the older stats shape, or before the second present); ' +
  'the figures summed across threads are not parts of the frame';

/// The phase names of these budgets, in the runtime's order, then any this version does not know.
function phaseNames(budgets) {
  const seen = new Set(budgets.flatMap((budget) => Object.keys(budget.phases)));
  return [...RUNTIME_PHASES.filter((name) => seen.has(name)), ...[...seen].filter((name) => !RUNTIME_PHASES.includes(name)).sort()];
}

/// Whether a budget's phases sum to its frame (`budget.frame_ms`) within the rounding of the figures written.
function budgetIdentity(budget) {
  const values = Object.values(budget.phases)
    .map((phase) => phase.ms)
    .filter(Number.isFinite);
  if (!Number.isFinite(budget.frameMs) || !values.length) return null;
  const sum = values.reduce((total, value) => total + value, 0);
  const tolerance = (budget.roundingMs ?? STATS_ROUNDING_MS) * (values.length + 1);
  return { sum, diff: sum - budget.frameMs, tolerance, holds: Math.abs(sum - budget.frameMs) <= tolerance + 1e-9 };
}

/// One budget's phases as `{ name: { ms, cpu_ms } }`, and whether they sum to its frame; nulls without a budget.
export function budgetParts(budget) {
  if (!budget || !Object.keys(budget.phases).length) return { parts: null, check: null };
  const parts = {};
  for (const name of phaseNames([budget])) parts[name] = { ms: budget.phases[name].ms, cpu_ms: budget.phases[name].cpuMs };
  const identity = budgetIdentity(budget);
  return {
    parts,
    check: identity ? { frame_ms: budget.frameMs, phases_sum_ms: round(identity.sum), holds: identity.holds } : null,
  };
}

/// The parts of the frame over several entries (`{ frames, budget }`): each budget phase's mean per frame, weighted
/// by the frames each budget covers, with the median and p95 of the entries' own means; its processor time where
/// every entry has one; and whether the phases sum to each budget's frame.
function partsSection(entries) {
  const rows = entries
    .filter((entry) => entry.budget && Object.keys(entry.budget.phases).length)
    .map((entry) => ({ budget: entry.budget, w: entry.budget.frames > 0 ? entry.budget.frames : entry.frames > 0 ? entry.frames : 1 }));
  if (!rows.length) return { parts: null, parts_note: PARTS_ABSENT_NOTE, budget: null };
  const parts = {};
  for (const name of phaseNames(rows.map((row) => row.budget))) {
    const having = rows.filter((row) => Number.isFinite(row.budget.phases[name]?.ms));
    if (!having.length) continue;
    const values = having.map((row) => row.budget.phases[name].ms);
    const cpu = having.every((row) => Number.isFinite(row.budget.phases[name].cpuMs))
      ? weightedMean(having.map((row) => ({ v: row.budget.phases[name].cpuMs, w: row.w })))
      : null;
    parts[name] = {
      mean: round(weightedMean(having.map((row) => ({ v: row.budget.phases[name].ms, w: row.w })))),
      cpu_ms: round(cpu),
      median: round(median(values)),
      p95: round(percentile(values, 0.95)),
      samples: having.length,
    };
  }
  const checks = rows.map((row) => ({ row, identity: budgetIdentity(row.budget) })).filter((check) => check.identity);
  const off = checks.filter((check) => !check.identity.holds);
  const worst = checks.reduce((max, check) => Math.max(max, Math.abs(check.identity.diff)), 0);
  const allowance = checks.reduce((max, check) => Math.max(max, check.identity.tolerance), 0);
  const holds = checks.length > 0 && off.length === 0;
  const budget = {
    frames: rows.reduce((sum, row) => sum + row.w, 0),
    frame_ms: round(weightedMean(rows.filter((row) => Number.isFinite(row.budget.frameMs)).map((row) => ({ v: row.budget.frameMs, w: row.w })))),
    phases_sum_ms: round(weightedMean(checks.map((check) => ({ v: check.identity.sum, w: check.row.w })))),
    objects: rows.length,
    identity: !checks.length ? 'n/a' : holds ? 'holds' : 'does not hold',
    objects_off: off.length,
    worst_diff_ms: round(worst),
    tolerance_ms: round(allowance, 4),
  };
  const parts_note = holds
    ? "parts: the presenting thread's budget phases, each a mean per frame (cpu_ms: its processor time, null where " +
      'not measured); they sum to budget.frame_ms within rounding, so they are the parts of the frame'
    : `parts: the presenting thread's budget phases; they do not sum to budget.frame_ms in ${off.length} of ` +
      `${checks.length} (worst ${round(worst)} ms off, rounding allows ${round(allowance, 4)} ms): they are not the ` +
      'whole frame there';
  return { parts, parts_note, budget };
}

/// Mean, median and p95 of every numeric leaf summed across threads, the mean weighted by frames when every entry
/// knows its frames.
function summedSection(entries, weighted) {
  const values = new Map();
  for (const entry of entries) {
    if (!entry.summed || typeof entry.summed !== 'object') continue;
    for (const [key, value] of Object.entries(flattenNumbers(entry.summed))) {
      if (!values.has(key)) values.set(key, []);
      values.get(key).push({ v: value, w: weighted ? entry.frames : 1 });
    }
  }
  const out = {};
  for (const [key, rows] of [...values.entries()].sort(([a], [b]) => a.localeCompare(b))) {
    const plain = rows.map((row) => row.v);
    out[key] = { mean: round(weightedMean(rows)), median: round(median(plain)), p95: round(percentile(plain, 0.95)), samples: rows.length };
  }
  return { summed_across_threads: out, summed_note: SUMMED_NOTE };
}

/// Frame time and its parts over stats records (readStatsObject) in order. The interval shape's mean is weighted by
/// the frames each object covers and its percentiles come from the object's own p50/p95 (one object) or the joined
/// frame_ms_list (several); the older sample's are the mean, median and p95 of its one-second figures.
export function summarizeObjects(records, { missed = 0 } = {}) {
  const figures = objectsFigures(records);
  const shapes = new Set(records.map((record) => record.shape));
  const weighted = figures.shape === 'interval';
  let frameMs;
  if (!weighted) {
    const described = describeValues(records.map((record) => record.frameMs).filter(Number.isFinite));
    frameMs = described ? { ...described, frames: null } : null;
  } else if (records.length === 1) {
    const [record] = records;
    frameMs = {
      mean: round(record.frameMs),
      median: round(record.frameMsP50 ?? figures.p50Ms),
      // An object's few frames have no 95th percentile of their own: the nearest rank of its list.
      p95: round(figures.p95Ms),
      samples: 1,
      frames: record.interval.frames,
      list_frames: figures.listFrames,
      list_missing: record.frameMsListMissing,
    };
  } else {
    const missing = records.reduce((sum, record) => sum + (record.frameMsListMissing ?? 0), 0);
    frameMs = {
      mean: round(figures.meanMs),
      median: round(figures.p50Ms),
      p95: round(figures.p95Ms),
      samples: figures.objects,
      frames: figures.frames,
      list_frames: figures.listFrames,
      list_missing: missing,
    };
  }
  const entries = records.map((record) => ({ frames: weighted ? record.interval.frames : null, budget: record.budget, summed: record.summed }));
  const first = records[0];
  const last = records[records.length - 1];
  return {
    stats_shape: shapes.size === 1 ? first.shape : 'mixed',
    samples: records.length,
    frames: weighted ? figures.frames : null,
    wall_ms: weighted && records.every((record) => Number.isFinite(record.interval.wallMs))
      ? round(records.reduce((sum, record) => sum + record.interval.wallMs, 0))
      : null,
    objects_missed: missed,
    frame: { first: first.interval?.firstFrame ?? first.frame, last: last.frame },
    frame_ms: frameMs,
    ...partsSection(entries),
    ...summedSection(entries, weighted),
  };
}

/// The stats file is rewritten about once a second. The server keeps its objects in memory so a window is more than
/// the single object on disk. In the interval shape an object is new when its seq is (a repeat is the same object
/// read again, whatever the file time says; a jump is objects missed; a lower seq is the runtime writing the file
/// afresh); in the older shape when the file's time moves.
class StatsWatcher {
  constructor(file, { intervalMs = 500, max = 1000, staleS = STALE_S, clock = REAL_CLOCK } = {}) {
    this.file = file;
    this.clock = clock;
    this.max = max;
    this.staleS = staleS;
    this.samples = [];
    this.latestRecord = null;
    this.lastMtimeMs = null;
    this.seqSince = null;
    this.cancel = null;
    if (file) {
      this.refresh();
      this.cancel = clock.every(intervalMs, () => this.refresh());
    }
  }

  refresh(now = this.clock.now()) {
    if (!this.file) return;
    let stat;
    try {
      stat = fs.statSync(this.file);
    } catch {
      return;
    }
    // The older shape has no seq: an unchanged file time is the same sample.
    if (this.latestRecord?.shape === 'sample' && stat.mtimeMs === this.lastMtimeMs) return;
    let record;
    try {
      record = readStatsObject(JSON.parse(fs.readFileSync(this.file, 'utf8')));
    } catch {
      return;
    }
    if (!record) return;
    this.lastMtimeMs = stat.mtimeMs;
    let missed = 0;
    if (record.shape === 'interval') {
      const seen = sequenceOf(this.latestRecord?.shape === 'interval' ? this.latestRecord : null, record);
      if (seen.kind === 'repeat') return;
      if (seen.kind === 'older') this.samples = [];
      missed = seen.missed;
      this.seqSince = now;
    } else {
      this.seqSince = null;
    }
    this.latestRecord = record;
    this.samples.push({ at: now, record, missed });
    if (this.samples.length > this.max) this.samples.splice(0, this.samples.length - this.max);
  }

  /// How old the newest object is and whether it is stale: by its own clock (unix_ms) or how long its seq has stood
  /// in the interval shape, by the file's time in the older one.
  freshness(now = this.clock.now()) {
    const record = this.latestRecord;
    if (!record) return null;
    const byClock = objectAgeMs(record, now);
    // A file's time is the real clock's, so it is aged on the real clock.
    const ageMs = byClock ?? (this.lastMtimeMs === null ? null : Date.now() - this.lastMtimeMs);
    const standingMs = record.seq !== null && this.seqSince !== null ? now - this.seqSince : null;
    const limitMs = (this.staleS[record.shape] ?? STALE_S.sample) * 1000;
    return {
      age_s: ageMs === null ? null : round(Math.max(0, ageMs) / 1000, 1),
      age_by: byClock !== null ? 'unix_ms' : ageMs !== null ? 'file_time' : null,
      seq_unchanged_s: standingMs === null ? null : round(standingMs / 1000, 1),
      stale: ageMs === null ? null : ageMs > limitMs || (standingMs !== null && standingMs > limitMs),
      limit_s: limitMs / 1000,
    };
  }

  /// The objects of the last `seconds`: in the interval shape by their own clock back from now, in the older one by
  /// when this server first saw each, back from the newest; never empty while there is a newest object.
  window(seconds, now = this.clock.now()) {
    this.refresh(now);
    if (!this.samples.length) return [];
    const last = this.samples[this.samples.length - 1];
    const within =
      last.record.unixMs !== null
        ? this.samples.filter((sample) => sample.record.unixMs !== null && sample.record.unixMs >= now - seconds * 1000)
        : this.samples.filter((sample) => sample.at >= last.at - seconds * 1000);
    return within.length ? within : [last];
  }

  close() {
    if (this.cancel) this.cancel();
    this.cancel = null;
  }
}

/// The words for a stale object's age.
function staleText(fresh, record) {
  const by = fresh.age_by === 'unix_ms' ? 'its own clock' : "the file's time";
  const seq = record.seq !== null ? ` (seq ${record.seq}, unchanged for ${fresh.seq_unchanged_s} s)` : '';
  return `the newest stats object${seq} is ${fresh.age_s} s old by ${by}, past the ${fresh.limit_s} s limit`;
}

function framePartsFromLog(logFile, seconds) {
  requireLog({ logFile });
  const text = readText(logFile) ?? '';
  const frames = scanFrames(text);
  if (!frames.length) {
    return {
      seconds, source: 'log', frame_line_shape: null, samples: 0, frame_ms: null, parts: null,
      summed_across_threads: null, note: 'no frame lines in the run log yet',
    };
  }
  let window = frames;
  const last = frames[frames.length - 1];
  if (last.tsMs !== null && frames.every((frame) => frame.tsMs !== null)) {
    const floor = last.tsMs - seconds * 1000;
    const within = frames.filter((frame) => frame.tsMs >= floor);
    if (within.length) window = within;
  } else {
    const rate = Number.isFinite(last.fps) && last.fps > 0 ? last.fps : 10;
    const count = Math.max(1, Math.min(frames.length, Math.round(seconds * rate)));
    window = frames.slice(-count);
  }
  // A line from job/timing-truth on says the frames its means cover: weight by them. An older line is one second's
  // sample: its figures are averaged as they are.
  const weighted = window.every((frame) => frame.window && frame.window.frames > 0 && Number.isFinite(frame.ms));
  let frameMs;
  let frameNote = null;
  if (!weighted) {
    const described = describeValues(window.map((frame) => frame.ms).filter(Number.isFinite));
    frameMs = described ? { ...described, frames: null } : null;
  } else {
    const only = window.length === 1 ? window[0] : null;
    frameMs = {
      mean: round(weightedMean(window.map((frame) => ({ v: frame.ms, w: frame.window.frames })))),
      median: only ? only.window.p50Ms : null,
      p95: only ? only.window.p95Ms : null,
      samples: window.length,
      frames: window.reduce((sum, frame) => sum + frame.window.frames, 0),
    };
    if (!only) {
      frameNote =
        "median and p95 are n/a over several frame lines: each line gives its own p50 and p95, not its frames' " +
        "times (the stats file's frame_ms_list has them)";
    }
  }
  const entries = window.map((frame) => ({ frames: weighted ? frame.window.frames : null, budget: frame.budget, summed: frame.summed }));
  return {
    seconds,
    source: 'log',
    frame_line_shape: last.window ? 'interval' : 'sample',
    samples: window.length,
    frames: weighted ? frameMs.frames : null,
    frame: { first: window[0].window?.firstFrame ?? window[0].frame, last: last.frame },
    frame_ms: frameMs,
    ...(frameNote ? { frame_ms_note: frameNote } : {}),
    ...partsSection(entries),
    ...summedSection(entries, weighted),
  };
}

// ----------------------------------------------------------------------------------------------------------------
// the control file

function assignmentsOf(text) {
  const out = {};
  for (const line of String(text ?? '').split(/\r?\n/)) {
    const assignment = parseAssignment(line);
    if (assignment && isLiveName(assignment.name)) out[assignment.name] = assignment.value;
  }
  return out;
}

function valueText(name, value, where) {
  if (typeof value !== 'string' && typeof value !== 'number' && typeof value !== 'boolean') {
    throw new ToolError(`${where}: ${name} must be a string, number or boolean`);
  }
  const text = (typeof value === 'boolean' ? (value ? 'on' : 'off') : String(value)).trim();
  try {
    checkValue(name, text);
  } catch (err) {
    throw new ToolError(err.message);
  }
  return text;
}

function normalizeSet(set) {
  if (!set || typeof set !== 'object' || Array.isArray(set)) {
    throw new ToolError('set must be an object of NAME: value');
  }
  const names = Object.keys(set);
  if (!names.length) throw new ToolError('set needs at least one switch');
  const out = {};
  for (const [name, raw] of Object.entries(set)) {
    try {
      checkName(name);
    } catch (err) {
      throw new ToolError(err.message);
    }
    if (!isLiveName(name)) {
      throw new ToolError(`'${name}' is not a live switch (live: ${LIVE_NAMES.join(', ')})`);
    }
    out[name] = valueText(name, raw, 'set');
  }
  return out;
}

function restoreControl(file, text) {
  try {
    if (text === null) fs.unlinkSync(file);
    else writeAtomic(file, text);
    return true;
  } catch {
    return false;
  }
}

/// The bytes appended to a log after `mark`, split into complete lines. The partial last line waits for the next
/// poll, as a writer may be midway through it.
class LogTail {
  constructor(file) {
    this.file = file;
    this.offset = 0;
    this.tail = '';
  }

  mark() {
    let stat;
    try {
      stat = fs.statSync(this.file);
    } catch (err) {
      throw new ToolError(`cannot read the run log (${errorCode(err)}); the game is not logging`);
    }
    this.offset = stat.size;
    this.tail = '';
  }

  readNew() {
    let stat;
    try {
      stat = fs.statSync(this.file);
    } catch (err) {
      throw new ToolError(`cannot read the run log (${errorCode(err)}); the game is not logging`);
    }
    if (stat.size < this.offset) {
      this.offset = 0;
      this.tail = '';
    }
    if (stat.size === this.offset) return [];
    const length = stat.size - this.offset;
    const buffer = Buffer.alloc(length);
    const fd = fs.openSync(this.file, 'r');
    try {
      fs.readSync(fd, buffer, 0, length, this.offset);
    } finally {
      fs.closeSync(fd);
    }
    this.offset = stat.size;
    const text = this.tail + buffer.toString('utf8');
    const lines = text.split('\n');
    this.tail = lines.pop();
    return lines;
  }
}

// ----------------------------------------------------------------------------------------------------------------
// tool implementations

function makeTools({ env, state, watcherRef, out, controller, clock }) {
  const sleep = clock.sleep;
  const markerTimeoutMs =
    Number(env.HARNESS_MCP_MARKER_TIMEOUT_MS) > 0 ? Number(env.HARNESS_MCP_MARKER_TIMEOUT_MS) : MARKER_TIMEOUT_MS;
  const runOptions = {
    markerTimeoutMs,
    stallMs: Number(env.HARNESS_MCP_STALL_MS) > 0 ? Number(env.HARNESS_MCP_STALL_MS) : undefined,
    pollMs: Number(env.HARNESS_MCP_POLL_MS) > 0 ? Number(env.HARNESS_MCP_POLL_MS) : undefined,
    holdScale: Number(env.HARNESS_MCP_HOLD_SCALE) > 0 ? Number(env.HARNESS_MCP_HOLD_SCALE) : 1,
    confirmTimeoutMs:
      Number(env.HARNESS_MCP_CONFIRM_TIMEOUT_MS) > 0 ? Number(env.HARNESS_MCP_CONFIRM_TIMEOUT_MS) : undefined,
  };
  const watcher = () => watcherRef.current;

  function run() {
    const where = resolveRun({ log: state.log, control: state.control, stats: state.stats, lock: state.lock }, env);
    // A session usually starts before the run does: follow the stats file when it appears (or goes away).
    if ((watcherRef.current?.file ?? null) !== (where.statsFile ?? null)) {
      if (watcherRef.current) watcherRef.current.close();
      watcherRef.current = new StatsWatcher(where.statsFile, { staleS: staleLimits(env), clock });
    }
    return where;
  }

  function gameStatus() {
    const where = run();
    let locked = false;
    let uptime = null;
    try {
      const stat = fs.statSync(where.lockFile);
      locked = true;
      let started = null;
      try {
        const owner = fs.readFileSync(path.join(where.lockFile, 'owner'), 'utf8');
        const match = owner.match(/\bstarted (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\b/);
        if (match) started = Date.parse(match[1].replace(' ', 'T'));
      } catch {
        // A lock without an owner file is still held; its folder's time is the best start there is.
      }
      const since = Number.isFinite(started) ? started : stat.mtimeMs;
      uptime = Math.max(0, Math.round((Date.now() - since) / 1000));
    } catch {
      locked = false;
    }

    const logText = where.logFile ? readText(where.logFile) : null;
    const header =
      logText === null
        ? { build: null, checkout: null, profile: null, console: null, switches: {} }
        : parseHeader(logText);
    const live = logText === null ? null : scanLive(logText);
    const latestSegment = live && live.segments.length ? live.segments[live.segments.length - 1] : null;
    const frames = logText === null ? [] : scanFrames(logText);
    const lastFrame = frames.length ? frames[frames.length - 1] : null;

    const statsWatcher = watcher();
    if (statsWatcher) statsWatcher.refresh();
    const record = statsWatcher ? statsWatcher.latestRecord : null;
    const fresh = record ? statsWatcher.freshness() : null;
    const stale = fresh?.stale === true;
    // A current object's figures; a stale one's are withheld (it stays under `stats`, as the last one written). Its
    // state (frame, switches, segments, video memory) is still the last known. Without a stats file, the log's last
    // frame line and the budget line after it.
    const current = record && !stale ? record : null;
    const line = record ? null : lastFrame;
    const figures = current
      ? {
          fps: current.fps,
          frameMs: current.frameMs,
          p50: current.frameMsP50,
          worst: current.frameMsWorst,
          interval: current.interval
            ? {
                first_frame: current.interval.firstFrame,
                last_frame: current.interval.lastFrame,
                frames: current.interval.frames,
                wall_ms: current.interval.wallMs,
              }
            : null,
          draws: current.draws,
          budget: current.budget,
          summed: current.summed,
        }
      : line
        ? {
            fps: line.fps,
            frameMs: line.ms,
            p50: line.window?.p50Ms ?? null,
            p95: line.window?.p95Ms ?? null,
            interval: line.window
              ? { first_frame: line.window.firstFrame, last_frame: line.window.lastFrame, frames: line.window.frames, wall_ms: line.window.wallMs }
              : null,
            draws: line.draws,
            budget: line.budget,
            summed: line.summed,
          }
        : null;
    const { parts, check } = budgetParts(figures?.budget ?? null);
    const open = record?.segmentOpen
      ? segmentOpenOut(record.segmentOpen, 'stats')
      : live?.open
        ? segmentOpenOut(live.open, 'log')
        : null;
    const absent = record?.shape === 'interval' ? OPTIONAL_PARTS.filter((name) => !record.optionalOn.includes(name)) : null;

    return {
      running: locked,
      locked,
      build: header.build ? redactLine(header.build) : null,
      checkout: header.checkout,
      profile: header.profile,
      console: header.console,
      switches: header.switches,
      uptime_s: uptime,
      source: record ? 'stats' : lastFrame ? 'log' : null,
      stats_shape: record ? record.shape : null,
      frame_line_shape: line ? (line.window ? 'interval' : 'sample') : null,
      seq: record ? record.seq : null,
      stats_age_s: fresh ? fresh.age_s : null,
      stats_age_by: fresh ? fresh.age_by : null,
      seq_unchanged_s: fresh ? fresh.seq_unchanged_s : null,
      stats_stale: fresh ? fresh.stale : null,
      ...(stale ? { stale_note: `${staleText(fresh, record)}: its figures are not current, so none are given; the object is under stats` } : {}),
      fps: figures ? figures.fps : null,
      frame_ms: figures ? figures.frameMs : null,
      frame_ms_p50: figures ? figures.p50 : null,
      // The stats object's longest frame (it has no 95th percentile); a frame line's window has a p95 of its own.
      ...(record ? { frame_ms_worst: figures ? (figures.worst ?? null) : null } : { frame_ms_p95: figures ? figures.p95 : null }),
      frame: record ? record.frame : lastFrame ? lastFrame.frame : null,
      interval: figures ? figures.interval : null,
      draws: figures ? figures.draws : null,
      // Per frame in the interval shape; the older shape's count was since the previous object.
      dispatched: current ? current.dispatched : null,
      dispatches_since_previous: current ? current.dispatchesSincePrevious : null,
      // The parts of the frame are the presenting thread's budget phases (diagnostics builds); the figures summed
      // across threads overlap each other and the frame and are not parts of it.
      parts,
      parts_check: check,
      summed_across_threads: safeValue(figures?.summed ?? null),
      vram_mb: record ? record.vramMb : null,
      live: safeValue(record?.live ?? live?.open?.inForce ?? latestSegment?.inForce ?? null),
      // The segments closed so far, by the stats object's name, else the log's last closed segment's number.
      segments_closed_total: record ? record.segmentsClosed : (latestSegment?.segment ?? null),
      segment_open: open,
      absent_optional: absent,
      ...(absent?.length ? { absent_optional_note: ABSENT_OPTIONAL_NOTE } : {}),
      kernel_calls: current?.kernelCalls ? safeValue({ per_frame: current.kernelCalls.perFrame }) : null,
      // The stats object is open-ended: a field this version does not know still reaches the caller.
      stats: record ? safeValue(record.raw) : null,
      log: logText !== null,
      control: where.controlFile ? existing(where.controlFile) : false,
    };
  }

  function frameParts(args = {}) {
    let seconds = args.seconds === undefined ? 10 : Number(args.seconds);
    if (!Number.isFinite(seconds) || seconds <= 0) throw new ToolError('seconds must be a positive number');
    seconds = Math.min(seconds, MAX_FRAME_WINDOW_S);
    const where = run();
    const statsWatcher = watcher();
    if (statsWatcher) {
      const now = clock.now();
      const samples = statsWatcher.window(seconds, now);
      const record = statsWatcher.latestRecord;
      if (record) {
        const fresh = statsWatcher.freshness(now);
        const age = { stats_age_s: fresh.age_s, stats_age_by: fresh.age_by, stats_stale: fresh.stale };
        if (fresh.stale) {
          // Not served as current, and not replaced by the log, whose last frame line is as old.
          return {
            seconds, source: 'stats', stats_shape: record.shape, samples: 0, frame_ms: null, parts: null,
            summed_across_threads: null, ...age, note: `no current figures: ${staleText(fresh, record)}`,
          };
        }
        const missed = samples.slice(1).reduce((sum, sample) => sum + sample.missed, 0);
        return { seconds, source: 'stats', ...summarizeObjects(samples.map((sample) => sample.record), { missed }), ...age };
      }
    }
    return framePartsFromLog(where.logFile, seconds);
  }

  function liveGet() {
    const where = run();
    const controlText = where.controlFile ? readControl(where.controlFile) : null;
    const requested = assignmentsOf(controlText);
    const logText = where.logFile ? readText(where.logFile) : null;
    const header = logText === null ? { switches: {} } : parseHeader(logText);
    const scanned = logText === null ? { markers: [], segments: [] } : scanLive(logText);
    const lastMarker = new Map();
    for (const marker of scanned.markers) lastMarker.set(marker.name, marker);
    const latestSegment = scanned.segments.length ? scanned.segments[scanned.segments.length - 1] : null;
    const statsWatcher = watcher();
    if (statsWatcher) statsWatcher.refresh();
    const statsRecord = statsWatcher?.latestRecord ?? null;
    const statsLive = statsRecord?.live ?? null;
    const fresh = statsRecord ? statsWatcher.freshness() : null;
    const stale = fresh?.stale === true;
    // In force now: the stats object's live map, else the open segment line (newer than the last closed one), else
    // the last closed segment's; each named by where it was read. A stale object's map is the last known, said so,
    // and acknowledges nothing.
    const inForce = statsLive ?? scanned.open?.inForce ?? latestSegment?.inForce ?? {};
    const forceSource = statsLive ? 'stats' : 'log';
    const asHeld = (name, value) => (value === null ? null : (normalizeLiveValue(name, value) ?? value));

    const names = LIVE_NAMES.map((name) => {
      const request = Object.hasOwn(requested, name) ? requested[name] : null;
      // The value asked as the runtime holds it (`1` is `on`), for the comparisons.
      const held = asHeld(name, request);
      const force = Object.hasOwn(inForce, name) ? String(inForce[name]) : null;
      const marker = lastMarker.get(name) ?? null;
      const markerMatches = request !== null && marker !== null && marker.new === held;
      let value = null;
      let source = null;
      if (markerMatches) {
        // A marker after the last segment is the newest applied value; it is also the acknowledgement.
        value = held;
        source = 'log';
      } else if (force !== null) {
        value = force;
        source = forceSource;
      } else if (request !== null) {
        value = request;
        source = 'control';
      } else if (Object.hasOwn(header.switches, name)) {
        value = header.switches[name];
        source = 'header';
      }
      const acknowledged = request === null || markerMatches || (!(stale && forceSource === 'stats') && force === held);
      return {
        name,
        kind: LIVE_KINDS[name] ?? 'text',
        value,
        requested: request,
        source,
        acknowledged,
        frame: markerMatches ? marker.frame : null,
      };
    });

    const pending = names.filter((entry) => !entry.acknowledged).map((entry) => entry.name);
    return {
      names,
      acknowledged: pending.length === 0,
      pending,
      control: { present: controlText !== null, assignments: Object.keys(requested).length },
      stats_stale: fresh ? fresh.stale : null,
      ...(stale && forceSource === 'stats'
        ? { stale_note: `${staleText(fresh, statsRecord)}: the values from it are the last known, not confirmed in force now` }
        : {}),
    };
  }

  async function liveSet(args = {}) {
    if (state.planRunning) throw new ToolError('a plan is running; live_set is refused until it finishes');
    if (state.switchBusy) throw new ToolError('another live switch write is in progress');
    const set = normalizeSet(args.set);
    const where = run();
    if (!where.controlFile) throw new ToolError('no control file: no live run is open');
    requireLog(where);
    const beforeText = readControl(where.controlFile);
    const before = assignmentsOf(beforeText);
    const changed = Object.keys(set).filter((name) => (before[name] ?? null) !== set[name]);
    const applied = Object.keys(set).map((name) => ({
      name,
      old: Object.hasOwn(before, name) ? before[name] : null,
      new: set[name],
      changed: changed.includes(name),
      frame: null,
    }));
    if (!changed.length) {
      return {
        applied,
        changed: false,
        acknowledged: true,
        waited_ms: 0,
        note: 'the control file already holds these values',
      };
    }

    const tail = new LogTail(where.logFile);
    tail.mark();
    state.switchBusy = true;
    const startedAt = clock.now();
    try {
      try {
        writeAtomic(where.controlFile, mergeControl(beforeText ?? '', { set }));
      } catch (err) {
        throw new ToolError(`cannot write the control file (${errorCode(err)})`);
      }

      const wanted = new Map(changed.map((name) => [name, set[name]]));
      const found = new Map();
      const deadline = startedAt + markerTimeoutMs;
      for (;;) {
        for (const line of tail.readNew()) {
          const parsed = parseLiveLine(line);
          if (!parsed?.change) continue;
          if (wanted.has(parsed.change.name) && parsed.change.new === wanted.get(parsed.change.name)) {
            found.set(parsed.change.name, parsed.change);
          }
        }
        if ([...wanted.keys()].every((name) => found.has(name))) break;
        const left = deadline - clock.now();
        if (left <= 0) {
          const missing = [...wanted.keys()].filter((name) => !found.has(name));
          const restored = restoreControl(where.controlFile, beforeText);
          throw new ToolError(
            `timed out after ${Math.round(markerTimeoutMs / 1000)} s waiting for the game to log ` +
              `${missing.join(', ')}; the game is not reading the control file` +
              (restored ? ' (the control file was restored)' : ' (the control file could not be restored)'),
            { restored },
          );
        }
        await sleep(Math.min(100, left));
      }

      for (const entry of applied) {
        const marker = found.get(entry.name);
        if (marker) {
          entry.frame = marker.frame;
          entry.old = marker.old;
        }
      }
      return { applied, changed: true, acknowledged: true, waited_ms: clock.now() - startedAt };
    } finally {
      state.switchBusy = false;
    }
  }

  /// The closed segments, either line shape, and the one under way last (`open` true), as live-report --json gives.
  function segments() {
    const where = run();
    requireLog(where);
    const text = readText(where.logFile) ?? '';
    const closed = parseLiveLog(text);
    const { open } = scanLive(text);
    if (!open) return closed;
    const previous = closed.length ? closed[closed.length - 1] : null;
    const delta = previous && previous.mean != null && open.mean != null ? Math.round((open.mean - previous.mean) * 10) / 10 : null;
    return [...closed, { ...open, meanChange: delta === 0 ? 0 : delta }];
  }

  function keysSequenceInfo(sequence) {
    if (typeof sequence !== 'string' || !sequence.trim()) {
      throw new ToolError('sequence must be a non-empty string of seconds:Key[:hold] entries');
    }
    try {
      checkValue('LIVE_KEYS', sequence);
    } catch (err) {
      throw new ToolError(err.message);
    }
    const entries = sequence
      .split(',')
      .map((entry) => entry.trim())
      .filter(Boolean);
    if (!entries.length) throw new ToolError('sequence has no presses');
    if (entries.length > MAX_KEYS_PRESSES) {
      throw new ToolError(`the sequence has ${entries.length} presses; the limit is ${MAX_KEYS_PRESSES}`);
    }
    let end = 0;
    for (const entry of entries) {
      const parts = entry.split(':');
      const at = Number(parts[0]);
      if (parts.length < 2 || !parts[1] || !Number.isFinite(at) || at < 0) {
        throw new ToolError(`'${entry}' is not seconds:Key[:hold]`);
      }
      const hold = parts.length >= 3 ? Number(parts[2]) : 0;
      const held = Number.isFinite(hold) && hold > 0 ? hold : 0;
      end = Math.max(end, at + held);
    }
    if (end > MAX_KEYS_S) {
      throw new ToolError(`the sequence is ${end} s long; the limit is ${MAX_KEYS_S} s`);
    }
    return { presses: entries.length, lengthMs: Math.round(end * 1000) };
  }

  async function sendKeys(args = {}) {
    if (!isLiveName('LIVE_KEYS')) {
      throw new ToolError('LIVE_KEYS is not available in this build');
    }
    if (state.planRunning) throw new ToolError('a plan is running; send_keys is refused until it finishes');
    if (state.switchBusy) throw new ToolError('another live switch write is in progress');
    const info = keysSequenceInfo(args.sequence);
    const where = run();
    if (!where.controlFile) throw new ToolError('no control file: no live run is open');
    requireLog(where);
    const beforeText = readControl(where.controlFile);
    const before = assignmentsOf(beforeText);
    if (before.LIVE_KEYS === args.sequence) {
      return {
        sequence: args.sequence,
        changed: false,
        presses: info.presses,
        length_s: round(info.lengthMs / 1000),
        note: 'the control file already holds this sequence; setting the same value again plays nothing',
      };
    }

    const tail = new LogTail(where.logFile);
    tail.mark();
    state.switchBusy = true;
    const startedAt = clock.now();
    const waitMs = info.lengthMs + 10_000;
    try {
      try {
        writeAtomic(where.controlFile, mergeControl(beforeText ?? '', { set: { LIVE_KEYS: args.sequence } }));
      } catch (err) {
        throw new ToolError(`cannot write the control file (${errorCode(err)})`);
      }
      let frame = null;
      let done = null;
      const deadline = startedAt + waitMs;
      for (;;) {
        for (const line of tail.readNew()) {
          const clean = stripAnsi(line).replace(/\r$/, '');
          const parsed = parseLiveLine(clean);
          if (parsed?.change && parsed.change.name === 'LIVE_KEYS') frame = parsed.change.frame;
          const finished = clean.match(/live keys: sequence done at frame (\d+), (\d+) presses/);
          if (finished) done = { frame: Number(finished[1]), presses: Number(finished[2]) };
        }
        if (done) break;
        const left = deadline - clock.now();
        if (left <= 0) {
          const restored = restoreControl(where.controlFile, beforeText);
          throw new ToolError(
            `timed out after ${round(waitMs / 1000)} s waiting for the runtime's "sequence done" line; ` +
              'the game is not playing live keys' +
              (restored ? ' (the control file was restored)' : ' (the control file could not be restored)'),
            { restored },
          );
        }
        await sleep(Math.min(100, left));
      }
      return {
        sequence: args.sequence,
        changed: true,
        presses: info.presses,
        length_s: round(info.lengthMs / 1000),
        frame,
        done_frame: done.frame,
        presses_played: done.presses,
        waited_ms: clock.now() - startedAt,
      };
    } finally {
      state.switchBusy = false;
    }
  }

  function rowsFor(logFile, timeline) {
    const text = readText(logFile);
    if (text === null) return [];
    const parsed = parseLiveLog(text);
    return reportRows(parsed, timeline ?? loadTimeline(logFile));
  }

  async function runPlanTool(args = {}) {
    if (state.planRunning) throw new ToolError('a plan is already running; run_plan is refused until it finishes');
    if (state.switchBusy) throw new ToolError('a live switch write is in progress; run_plan is refused until it finishes');
    let plan;
    try {
      plan = normalizePlan(args.plan);
    } catch (err) {
      throw new ToolError(err.message);
    }
    const where = run();
    if (!where.controlFile) throw new ToolError('no control file: no live run is open');
    requireLog(where);
    if (!where.statsFile) {
      throw new ToolError("no stats file: a plan needs the switches in force from the run's stats.json (a --live run)");
    }

    state.planRunning = true;
    const startedAt = new Date().toISOString();
    try {
      const result = await runPlan(plan, {
        logFile: where.logFile,
        controlFile: where.controlFile,
        statsFile: where.statsFile,
        markerTimeoutMs: runOptions.markerTimeoutMs,
        ...(runOptions.confirmTimeoutMs ? { confirmTimeoutMs: runOptions.confirmTimeoutMs } : {}),
        ...(runOptions.stallMs ? { stallMs: runOptions.stallMs } : {}),
        ...(runOptions.pollMs ? { pollMs: runOptions.pollMs } : {}),
        holdScale: runOptions.holdScale,
        ...(clock === REAL_CLOCK ? {} : { now: clock.now, sleep: clock.sleep }),
        signal: controller.signal,
        out,
        err: out,
        writeTimeline: true,
      });
      const rows = rowsFor(where.logFile, result.timeline);
      return {
        plan: result.timeline.plan,
        started_at: startedAt,
        finished_at: result.timeline.finishedAt,
        restored: result.timeline.restored,
        restore_note: result.timeline.restoreNote,
        stopped: result.timeline.stopped,
        rows,
      };
    } catch (err) {
      let rows = null;
      try {
        rows = rowsFor(where.logFile, err.timeline ?? null);
      } catch {
        rows = null;
      }
      // A refusal wrote nothing; a stop carries the timeline, which says whether the baseline was confirmed.
      const restored = err.timeline ? err.timeline.restored : true;
      const note = err.timeline ? err.timeline.restoreNote : 'nothing was written';
      throw new ToolError(err.message, { rows, restored, restore_note: note });
    } finally {
      state.planRunning = false;
    }
  }

  function compilePattern(pattern) {
    if (typeof pattern !== 'string' || !pattern) throw new ToolError('pattern must be a non-empty string');
    if (pattern.length > 200) throw new ToolError('pattern is too long (at most 200 characters)');
    if (pattern.startsWith('/') && pattern.length > 2) {
      const end = pattern.lastIndexOf('/');
      if (end > 0) {
        const source = pattern.slice(1, end);
        const flags = pattern.slice(end + 1);
        if (!/^[a-z]*$/.test(flags) || /[gy]/.test(flags)) {
          throw new ToolError('a /pattern/ may use only the i, m, s and u flags');
        }
        if (source.length > 120) throw new ToolError('the regular expression is too long (at most 120 characters)');
        if (/\\[1-9]|\(\?<[=!]|(?:\([^)]*[+*][^)]*\))[+*]/.test(source)) {
          throw new ToolError('that regular expression is not a small safe pattern');
        }
        try {
          const re = new RegExp(source, flags);
          return (line) => re.test(line);
        } catch (err) {
          throw new ToolError(`bad regular expression: ${err.message}`);
        }
      }
    }
    return (line) => line.includes(pattern);
  }

  function logFind(args = {}) {
    const where = run();
    requireLog(where);
    const matches = compilePattern(args.pattern);
    let last = args.last === undefined ? 20 : Number(args.last);
    if (!Number.isFinite(last) || last < 1) last = 20;
    last = Math.min(Math.floor(last), MAX_LOG_FIND);
    const text = readText(where.logFile) ?? '';
    const lines = [];
    let total = 0;
    let withheld = 0;
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.replace(/\r$/, '');
      if (!matches(line)) continue;
      // A line holding a web address is never returned (web-address.mjs): counted, not masked.
      if (withoutWebAddresses([line]).withheld) {
        withheld += 1;
        continue;
      }
      total += 1;
      lines.push(redactLine(line));
      if (lines.length > last) lines.shift();
    }
    return { pattern: redactLine(args.pattern), total, lines, withheld, ...(withheld ? { withheld_note: withheldText(withheld) } : {}) };
  }

  return { gameStatus, frameParts, liveGet, liveSet, segments, runPlanTool, sendKeys, logFind };
}

// ----------------------------------------------------------------------------------------------------------------
// MCP

const TOOL_DEFINITIONS = Object.freeze([
  {
    name: 'game_status',
    description:
      'Use to see whether a live run holds the lock, its build, console and switches, and the latest fps, frame time, parts and video memory.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'frame_parts',
    description:
      'Use to get the mean, median and 95th percentile of frame time and each part over the last N seconds (default 10, at most 120).',
    inputSchema: {
      type: 'object',
      properties: {
        seconds: { type: 'number', minimum: 1, maximum: 120, default: 10, description: 'Window length in seconds.' },
      },
      additionalProperties: false,
    },
  },
  {
    name: 'live_get',
    description:
      'Use to read every live switch, its kind and current value, and whether the runtime has acknowledged the latest change.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'live_set',
    description:
      'Use to apply one or more live switches between frames and wait for the runtime markers; refuses bad names and values and restores the file if nothing answers.',
    inputSchema: {
      type: 'object',
      properties: {
        set: {
          type: 'object',
          minProperties: 1,
          additionalProperties: { type: ['string', 'number', 'boolean'] },
          description: 'Switch names and their new values.',
        },
      },
      required: ['set'],
      additionalProperties: false,
    },
  },
  {
    name: 'segments',
    description:
      'Use to read the live segment table from the run log as objects, newest last, with short stretches marked.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
  },
  {
    name: 'run_plan',
    description:
      'Use to run an A/B/A experiment plan through live-drive, with its stop rules, and return the verdict table; refuses a plan over the driver limits or while another runs.',
    inputSchema: {
      type: 'object',
      properties: {
        plan: {
          type: 'object',
          description: 'live-drive format: {baseline, steps: [{label, set, hold_s}], repeat}.',
        },
      },
      required: ['plan'],
      additionalProperties: false,
    },
  },
  {
    name: 'send_keys',
    description:
      'Use to play a timed seconds:Key sequence in the running title through LIVE_KEYS and wait for its sequence-done line.',
    inputSchema: {
      type: 'object',
      properties: {
        sequence: { type: 'string', minLength: 1, description: 'Comma-separated seconds:Key[:hold] entries.' },
      },
      required: ['sequence'],
      additionalProperties: false,
    },
  },
  {
    name: 'log_find',
    description:
      'Use to search the run log for a warning or counter the other tools do not cover; private paths and 16-hex ids are redacted.',
    inputSchema: {
      type: 'object',
      properties: {
        pattern: {
          type: 'string',
          minLength: 1,
          description: 'A plain substring, or /pattern/ for a small safe regex.',
        },
        last: {
          type: 'integer',
          minimum: 1,
          maximum: 40,
          default: 20,
          description: 'Return at most this many matching lines.',
        },
      },
      required: ['pattern'],
      additionalProperties: false,
    },
  },
]);

function toolResult(value, isError = false) {
  return { content: [{ type: 'text', text: JSON.stringify(value) }], isError };
}

function toolFailure(err) {
  const detail = err instanceof ToolError ? err.detail : {};
  const message = redactLine(err && err.message ? err.message : String(err));
  return toolResult({ error: message, ...detail }, true);
}

export function createProtocol({ env = process.env, out = process.stderr, clock = REAL_CLOCK } = {}) {
  const state = { planRunning: false, switchBusy: false, log: null, control: null, stats: null, lock: null };
  const controller = new AbortController();
  const watcherRef = { current: null };
  const refreshWatcher = () => {
    const where = resolveRun(state, env);
    if (watcherRef.current && watcherRef.current.file === where.statsFile) return;
    if (watcherRef.current) watcherRef.current.close();
    watcherRef.current = new StatsWatcher(where.statsFile, { staleS: staleLimits(env), clock });
  };
  refreshWatcher();
  const tools = makeTools({ env, state, watcherRef, out, controller, clock });

  function setRunFiles({ log, control, stats, lock } = {}) {
    state.log = log ?? null;
    state.control = control ?? null;
    state.stats = stats ?? null;
    state.lock = lock ?? null;
    refreshWatcher();
  }

  function toolByName(name) {
    switch (name) {
      case 'game_status':
        return () => tools.gameStatus();
      case 'frame_parts':
        return (args) => tools.frameParts(args);
      case 'live_get':
        return () => tools.liveGet();
      case 'live_set':
        return (args) => tools.liveSet(args);
      case 'segments':
        return () => tools.segments();
      case 'run_plan':
        return (args) => tools.runPlanTool(args);
      case 'send_keys':
        return (args) => tools.sendKeys(args);
      case 'log_find':
        return (args) => tools.logFind(args);
      default:
        return null;
    }
  }

  async function handle(message) {
    if (
      !message ||
      typeof message !== 'object' ||
      Array.isArray(message) ||
      typeof message.method !== 'string'
    ) {
      if (message && typeof message === 'object' && !('id' in message)) return null;
      return {
        jsonrpc: '2.0',
        id: message && typeof message === 'object' && 'id' in message ? message.id : null,
        error: { code: -32600, message: 'Invalid Request' },
      };
    }
    if (!('id' in message)) return null; // a notification, initialized included: no answer

    const id = message.id;
    const params = message.params && typeof message.params === 'object' ? message.params : {};

    if (message.method === 'initialize') {
      return {
        jsonrpc: '2.0',
        id,
        result: {
          protocolVersion: typeof params.protocolVersion === 'string' ? params.protocolVersion : '2025-06-18',
          capabilities: { tools: {} },
          serverInfo: SERVER_INFO,
          instructions: INSTRUCTIONS,
        },
      };
    }
    if (message.method === 'ping') return { jsonrpc: '2.0', id, result: {} };
    if (message.method === 'tools/list') return { jsonrpc: '2.0', id, result: { tools: TOOL_DEFINITIONS } };
    if (message.method === 'tools/call') {
      if (typeof params.name !== 'string') {
        return { jsonrpc: '2.0', id, result: toolResult({ error: 'tools/call needs a tool name' }, true) };
      }
      const handler = toolByName(params.name);
      if (!handler) {
        return { jsonrpc: '2.0', id, result: toolResult({ error: `unknown tool '${params.name}'` }, true) };
      }
      try {
        const args = params.arguments && typeof params.arguments === 'object' ? params.arguments : {};
        return { jsonrpc: '2.0', id, result: toolResult(await handler(args)) };
      } catch (err) {
        return { jsonrpc: '2.0', id, result: toolFailure(err) };
      }
    }
    return { jsonrpc: '2.0', id, error: { code: -32601, message: `Method not found: ${message.method}` } };
  }

  function close() {
    // A client that goes away mid-plan gets the baseline restored, as an interrupt would.
    controller.abort();
    if (watcherRef.current) watcherRef.current.close();
    watcherRef.current = null;
  }

  return { handle, setRunFiles, close };
}

export function parseArgs(argv, env = process.env) {
  const options = { log: null, control: null, stats: null, lock: null, help: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--log' || arg === '--file' || arg === '--stats' || arg === '--lock') {
      if (i + 1 >= argv.length) {
        const err = new Error(`${arg} needs a path`);
        err.exitCode = 2;
        throw err;
      }
      options[{ '--log': 'log', '--file': 'control', '--stats': 'stats', '--lock': 'lock' }[arg]] = argv[(i += 1)];
    } else if (arg === '--help' || arg === '-h') {
      options.help = true;
    } else {
      const err = new Error(`unknown argument '${arg}'`);
      err.exitCode = 2;
      throw err;
    }
  }
  if (!options.log) options.log = env.HARNESS_LOG || env.HARNESS_RUN_LOG || null;
  if (!options.control) options.control = env.VICTORY_LIVE || env.HARNESS_LIVE || null;
  if (!options.stats) options.stats = env.VICTORY_LIVE_STATS || env.HARNESS_LIVE_STATS || null;
  return options;
}

const USAGE =
  'usage: node harness/mcp/live-server.mjs [--log <run log>] [--file <control file>] [--stats <stats file>]\n' +
  '                                      [--lock <run lock>]\n';

export function main(argv = process.argv.slice(2), env = process.env) {
  let options;
  try {
    options = parseArgs(argv, env);
  } catch (err) {
    process.stderr.write(`live-mcp: ${redactLine(err.message)}\n${USAGE}`);
    return Promise.resolve(err.exitCode || 2);
  }
  if (options.help) {
    process.stderr.write(USAGE);
    return Promise.resolve(0);
  }

  const protocol = createProtocol({ env });
  protocol.setRunFiles(options);
  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

  const send = (message) => {
    if (!message) return;
    try {
      process.stdout.write(`${JSON.stringify(message)}\n`);
    } catch {
      // A closed pipe ends the session; there is nothing useful to say on it.
    }
  };

  rl.on('line', (line) => {
    if (!line.trim()) return;
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      send({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'Parse error' } });
      return;
    }
    // Requests are handled independently: a long run_plan must not block a ping or a status read.
    Promise.resolve(protocol.handle(message)).then(send, (err) => {
      process.stderr.write(`live-mcp: ${redactLine(err && err.message ? err.message : String(err))}\n`);
      if (message && typeof message === 'object' && 'id' in message) {
        send({ jsonrpc: '2.0', id: message.id, error: { code: -32603, message: 'Internal error' } });
      }
    });
  });
  rl.on('close', () => protocol.close());

  return new Promise((resolve) => {
    rl.on('close', () => resolve(0));
  });
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  main().then(
    (code) => {
      process.exitCode = code;
    },
    (err) => {
      process.stderr.write(`live-mcp: ${redactLine(err && err.message ? err.message : String(err))}\n`);
      process.exitCode = 1;
    },
  );
}
