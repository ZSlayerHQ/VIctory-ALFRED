#!/usr/bin/env node
// Drive an open live session from a plan, then read the arms back.
//
//   live-drive.sh run <plan.json> --log <run log> [--file <control file>] [--stats <stats file>]
//   live-drive.sh report --log <run log> [--plan <n>] [--json]
//
// A plan is {"baseline": {NAME: value, ...}, "steps": [{"label": "...", "set": {NAME: value, ...}, "hold_s": 30,
// "settle_s": 5}], "repeat": 2, "settle_s": 0}. The runner holds the baseline once (the first baseline arm), then for
// every step: sets the switches, holds hold_s seconds (the step arm), restores the baseline and holds hold_s more (the
// next baseline arm). So every step is bracketed A, B, A. `repeat` (default 2) counts the passes: the second runs the
// steps in reverse order, which cancels an order effect; a third runs forward again. `settle_s` (default 0, at most
// half of hold_s) is the seconds at the start of an arm whose samples are left out as settling, for a switch whose
// effect builds up after it applies: a step's own value (else the plan's) covers its step arm and the baseline arm
// after it; the plan's covers the first baseline arm. A step whose state equals the baseline (start values included)
// is refused.
//
// One runner per control file: it takes a lock beside it (<control>.live-drive.lock: process id, start time) and is
// refused while a process holding it exists; a lock whose process is gone is stale and taken over. Before its first
// write it leaves a state file beside the control file (<control>.live-drive-state.json: the start values, the
// sidecar, the process id, the start time) and removes it once the final baseline is confirmed. A runner killed
// mid-arm leaves its step values in force and that file behind: a plan that finds it refuses to start until the stats
// file shows the recorded start values, and names the switches that differ and the command that restores them.
//
// The runtime applies only the names PRESENT in the control file (`victory-gpu/src/live.rs`, `apply`): a name
// removed from the file keeps its last value. So the runner never unsets a name. Before it writes anything it reads
// the switches in force from the stats file's `live` map (`host/stats.rs`) and refuses to start when that map is
// missing or the file is over 15 s old; every state it writes names every switch the plan touches, with the
// plan's value or, where the baseline does not name it, the start value. The first write also sets `LIVE_MARK` to a
// number the runtime has not seen (where its live map has the switch): a mark changes nothing and ends the segment
// under way, so the stretch before the plan, its loading with it, is a segment of its own; a plan may not set it.
// After every write it waits for the
// runtime's `live: <NAME> ...` markers of the names that change (15 s), then for a stats sample newer than the write
// whose `live` map shows exactly the values asked (20 s); only then does an arm's clock start. After the plan (on
// success, a stop or SIGINT/SIGTERM) it writes the baseline and confirms it the same way, and the sidecar and the
// last line say plainly when it could not.
//
// An arm is measured from the stats samples read during its hold, so it starts after the plan did: no loading or
// intro from before the plan is in the first baseline. A sample whose frame did not advance (`stalled`: the older
// shape, which rewrote its object while nothing presented), a stats file unchanged for over 15 s (`stale`, counted
// once per 15 s) and a read that failed or did not parse (`unparsable`) are counted and left out. Every sample keeps
// the whole `live` map and the stats file's count of closed segments (`segments_closed_total`, kept as
// `segments_closed`; a version-2 sidecar written before this kept it as `segment`), so the report sees a switch
// flipped by hand.
//
// The runner stops early when the log stops growing for 20 s, when a line carries `device removed` or `DXGI_ERROR`,
// or when a closed log segment's mean, or the open one's once it has counted 10 frames, is over five times
// the first closed baseline segment's (a segment with nothing counted has no mean and is skipped). It refuses an
// unknown live name, `LIVE_MARK` (the runner's own), a
// value the runtime would refuse, more than 40 steps and a plan over 45 minutes.
//
// The labels are not in the game's log, so every plan writes its own sidecar beside the log, claimed when it starts so
// that no plan replaces another's: the first <log>.live-drive.json, the next <log>.live-drive.2.json, and so on
// (version 2: the plan, the start values, every arm's target and samples, the `live:` changes the log showed during
// the plan, and the log segments each label produced, for the picture check and the p95). `report --log` gives one
// table per plan, in order; `--plan <n>` gives one; `--json` gives every row with its plan's number.
//
// The report compares arm medians: with the interval shape the median of the arm's joined `frame_ms_list`s, with the
// older shape the median of its samples' `frame_ms`. First a frame (or, without frame lists, a stats object) over 5
// times its arm's median is left out as a pause or hitch and named: the runtime writes nothing while nothing presents,
// so a pause arrives as one object holding the whole gap. A pass's effect is its step arm's median minus the mean of
// its two baseline arms' medians; the difference is the mean of the passes' effects; `needed` is the larger of 3% of
// the baseline and twice the spread (the range of the passes' effects or of the baseline arms' medians, the larger),
// the smallest effect the plan could have told from its own noise, given in ms and %. Every label gets one verdict:
//   unreliable            an arm's samples show values other than the plan's (the target) or a value that changed
//                         mid-arm; the log shows a `live:` change inside an arm, or the stats file's closed-segment
//                         count moved within one (a switch flipped and flipped back between two samples); or an arm
//                         the plan never measured; also every row of an older sidecar or of a log without one, since
//                         those runs restored by removing names and never confirmed what was in force;
//   too few samples       an arm has fewer than 3 good samples or they span fewer than 30 frames: no figures;
//   one pass: no verdict  one pass has nothing to measure its own noise by: its figures, never a direction;
//   no clear change       the passes disagree in sign, or |difference| is not over `needed`. It is not evidence of no
//                         effect: an effect under `needed` cannot be told from this plan's noise;
//   faster, slower        otherwise.
//
// There is no `keys` subcommand: run-world.sh passes VICTORY_KEYS in the environment when the title starts, and the
// runtime's LIVE_KEYS switch (send_keys in the MCP server) plays a sequence in a running title.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { checkName, checkValue, mergeControl, parseAssignment, readControl, writeAtomic } from './live.mjs';
import { LIVE_NAMES, UNLISTED_WHEN_OFF, isLiveName, normalizeLiveValue } from './live-names.mjs';
import { parseLiveLine, parseLiveLog } from './live-report.mjs';
import { COULD_NOT_CHECK, FINDINGS, annotatePictureSegments } from './picture-check.mjs';
import { stripAnsi } from './parse-run.mjs';
import { objectAgeMs, objectsFigures, parseStatsText, sequenceOf } from './stats-object.mjs';

export const MIN_HOLD_S = 10;
export const MAX_HOLD_S = 300;
export const MAX_STEPS = 40;
export const MAX_TOTAL_S = 45 * 60;
export const MARKER_TIMEOUT_MS = 15_000;
/// How long after a write the runner waits for a stats sample that shows it. The runtime writes the stats file
/// about once in six seconds (the frame-rate line's five-second gate), so this is three of its periods.
export const CONFIRM_TIMEOUT_MS = 20_000;
/// A stats file older than this at the start is refused; during an arm, this long without a rewrite counts as stale.
export const STATS_STALE_MS = 15_000;
export const STALL_MS = 20_000;
export const POLL_MS = 250;
export const FAILURE_RE = /DXGI_ERROR|device[ _]removed/i;
/// An arm's figures need at least this many good samples, spanning at least this many presented frames.
export const MIN_GOOD_SAMPLES = 3;
export const MIN_ARM_FRAMES = 30;
/// faster/slower needs |difference| over the larger of this share of the baseline and SPREAD_FACTOR x the spread.
export const MIN_EFFECT_PERCENT = 3;
export const SPREAD_FACTOR = 2;
/// A frame (or, without frame lists, a stats object) over this many times its arm's median is a pause or hitch.
export const PAUSE_FACTOR = 5;
/// The passes a plan runs when it gives no `repeat`; one pass never gives a direction.
export const DEFAULT_REPEAT = 2;
/// The settling seconds left out at an arm's start when a plan gives no `settle_s`.
export const DEFAULT_SETTLE_S = 0;
export const ONE_PASS = 'one pass: no verdict';

/// The rules a report applies, said once under its table.
export const RULES_TEXT =
  `rules: figures are arm medians (of the joined frame lists, else of the samples' frame_ms), after a frame or ` +
  `object over ${PAUSE_FACTOR} x its arm's median is left out as a pause or hitch, and samples in an arm's first ` +
  `settle_s as settling; an arm needs at least ${MIN_GOOD_SAMPLES} good stats samples spanning ${MIN_ARM_FRAMES} ` +
  `frames; samples whose frame did not advance, a stats file unchanged for over ${STATS_STALE_MS / 1000} s and reads ` +
  'that failed or did not parse are left out and counted; an arm whose samples show values other than the plan asked ' +
  "or a value that changed, or in which the log shows a change or the stats file's segment count moved, makes its " +
  "row unreliable; a pass's effect is its step arm's median minus the mean of its two baseline arms'; one pass gives " +
  'no verdict; faster or slower needs every pass to agree in sign and |difference| over needed, the larger of ' +
  `${MIN_EFFECT_PERCENT}% of the baseline and ${SPREAD_FACTOR} x the spread (the range of the passes' effects or of ` +
  "the baseline arms' medians), else no clear change. no clear change is not evidence of no effect: an effect under " +
  'needed cannot be told from this plan\'s noise.';

const NA = 'n/a';
// Findings outrank could-not-check, which outranks clean; picture-check's own lists, in its order.
const pictureRank = (verdict) =>
  FINDINGS.includes(verdict) ? 1 + FINDINGS.indexOf(verdict) : COULD_NOT_CHECK.includes(verdict) ? 0.5 : 0;

/// A plan or command-line fault: the runner stops before touching the control file, and the shell gets exit 2.
export class PlanError extends Error {
  constructor(message) {
    super(message);
    this.exitCode = 2;
  }
}

/// A reason the run stopped: the log stalled, a device was removed, a stretch broke, a marker or a confirmation
/// never came, or the operator interrupted. The baseline is written and confirmed (or said not to be) first.
export class StopError extends Error {
  constructor(message, exitCode = 1) {
    super(message);
    this.exitCode = exitCode;
  }
}

const realSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const average = (values) => (values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null);
const range = (values) => (values.length > 1 ? Math.max(...values) - Math.min(...values) : 0);
const round1 = (value) => (Number.isFinite(value) ? Math.round(value * 10) / 10 : null);
const one = (value) => (Number.isFinite(value) ? value.toFixed(1) : NA);
const signed = (value) => (Number.isFinite(value) ? `${value > 0 ? '+' : ''}${value.toFixed(1)}` : NA);
const seconds = (ms) =>
  ms >= 1000 ? `${(ms / 1000).toFixed(ms % 1000 === 0 ? 0 : 1)} s` : `${ms} ms`;
const cell = (value) => String(value).replace(/\|/g, '\\|');
const valueShown = (value) => (value === '' ? '(empty)' : value);
const setText = (set) =>
  Object.entries(set)
    .map(([name, value]) => `${name}=${valueShown(value)}`)
    .join(' ');

/// A JSON value as the control file spells it: strings stay, numbers print, booleans become on/off. The value may
/// not carry a newline or `#`, the same rule live.sh applies, and must be one the runtime accepts for its kind.
function valueText(name, value, where) {
  if (typeof value !== 'string' && typeof value !== 'number' && typeof value !== 'boolean') {
    throw new PlanError(`${where}: ${name} must be a string, number or boolean`);
  }
  const text = (typeof value === 'boolean' ? (value ? 'on' : 'off') : String(value)).trim();
  checkValue(name, text);
  if (normalizeLiveValue(name, text) === null) {
    throw new PlanError(`${where}: ${name} must be a whole number; the runtime would keep its old value`);
  }
  return text;
}

function switchMap(value, where, { nonEmpty = false } = {}) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new PlanError(`${where} must be an object of NAME: value`);
  }
  const out = {};
  for (const [name, raw] of Object.entries(value)) {
    checkName(name);
    if (!isLiveName(name)) {
      throw new PlanError(`${where}: '${name}' is not a live switch (live: ${LIVE_NAMES.join(', ')})`);
    }
    if (name === MARK) {
      throw new PlanError(`${where}: ${MARK} is the runner's own: it writes a new number at the plan's start, and a plan may not set it`);
    }
    out[name] = valueText(name, raw, where);
  }
  if (nonEmpty && !Object.keys(out).length) throw new PlanError(`${where} needs at least one switch`);
  return out;
}

/// A settle_s as given (null when absent): a number of seconds, 0 or more.
function settleOf(value, where) {
  if (value === undefined || value === null) return null;
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new PlanError(`${where}: settle_s must be a number of seconds, 0 or more`);
  }
  return value;
}

const sameValue = (name, a, b) => (normalizeLiveValue(name, a) ?? a) === (normalizeLiveValue(name, b) ?? b);

/// The plan with its values as strings and every rule checked; every step carries the settle_s it uses. Throws
/// PlanError (exit 2).
export function normalizePlan(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw new PlanError('the plan must be a JSON object');
  }
  const baseline = input.baseline === undefined ? {} : switchMap(input.baseline, 'baseline');
  const planSettle = settleOf(input.settle_s, 'the plan') ?? DEFAULT_SETTLE_S;
  if (!Array.isArray(input.steps) || input.steps.length === 0) {
    throw new PlanError('the plan needs a non-empty steps array');
  }
  if (input.steps.length > MAX_STEPS) {
    throw new PlanError(`the plan has ${input.steps.length} steps; the limit is ${MAX_STEPS}`);
  }
  const labels = new Set();
  const steps = input.steps.map((step, index) => {
    const where = `step ${index + 1}`;
    if (!step || typeof step !== 'object' || Array.isArray(step)) {
      throw new PlanError(`${where} must be an object`);
    }
    if (typeof step.label !== 'string' || !step.label.trim()) {
      throw new PlanError(`${where} needs a non-empty label`);
    }
    const label = step.label.trim();
    if (/[\r\n]/.test(label)) {
      throw new PlanError(`${where}: the label may not contain a newline`);
    }
    if (labels.has(label)) {
      throw new PlanError(`${where}: duplicate label '${label}'; a label names one report row`);
    }
    labels.add(label);
    const set = switchMap(step.set, `${where} ('${label}') set`, { nonEmpty: true });
    if (typeof step.hold_s !== 'number' || !Number.isFinite(step.hold_s)) {
      throw new PlanError(`${where} ('${label}'): hold_s must be a number of seconds`);
    }
    if (step.hold_s < MIN_HOLD_S || step.hold_s > MAX_HOLD_S) {
      throw new PlanError(
        `${where} ('${label}'): hold_s is ${step.hold_s}; it must be between ${MIN_HOLD_S} and ${MAX_HOLD_S} seconds`,
      );
    }
    const settle = settleOf(step.settle_s, `${where} ('${label}')`) ?? planSettle;
    if (settle > step.hold_s / 2) {
      throw new PlanError(
        `${where} ('${label}'): settle_s is ${settle}; it may be at most half of hold_s (${step.hold_s / 2} s)`,
      );
    }
    if (Object.entries(set).every(([name, value]) => Object.hasOwn(baseline, name) && sameValue(name, baseline[name], value))) {
      throw new PlanError(`${where} ('${label}'): its set equals the baseline, so its arms would compare the baseline with itself`);
    }
    return { label, set, hold_s: step.hold_s, settle_s: settle };
  });
  if (planSettle > steps[0].hold_s / 2) {
    throw new PlanError(
      `the plan's settle_s is ${planSettle}; the first baseline arm holds ${steps[0].hold_s} s, so it may be at most ` +
        `${steps[0].hold_s / 2} s`,
    );
  }
  const repeat = input.repeat === undefined ? DEFAULT_REPEAT : input.repeat;
  if (!Number.isInteger(repeat) || repeat < 1) {
    throw new PlanError('repeat must be a positive integer (1 runs the list once, and gives no verdict)');
  }
  const plan = { baseline, steps, repeat, settle_s: planSettle };
  const total = totalSeconds(plan);
  if (total > MAX_TOTAL_S) {
    throw new PlanError(
      `the plan is ${(total / 60).toFixed(1)} minutes long; the limit is ${MAX_TOTAL_S / 60} minutes ` +
        '(the initial baseline and both holds for every step of every pass)',
    );
  }
  return plan;
}

/// Every step occurrence in run order: one pass forward, the next in reverse, and so on. `pass` is 1-based.
export function occurrencesOf(plan) {
  const out = [];
  for (let pass = 0; pass < plan.repeat; pass += 1) {
    const indices = plan.steps.map((_, index) => index);
    const reversed = pass % 2 === 1;
    if (reversed) indices.reverse();
    for (const index of indices) {
      const step = plan.steps[index];
      out.push({
        pass: pass + 1,
        direction: reversed ? 'reverse' : 'forward',
        index,
        label: step.label,
        set: step.set,
        hold_s: step.hold_s,
        settle_s: step.settle_s ?? plan.settle_s ?? DEFAULT_SETTLE_S,
      });
    }
  }
  return out;
}

/// The holds the plan asks for: the initial baseline, then a step hold and a baseline hold for every occurrence.
/// Marker and confirmation waits are not included; they are bounded by their timeouts. Arithmetic, not a list of
/// occurrences, so a huge repeat is refused instead of built.
export function totalSeconds(plan) {
  const initial = plan.steps[0]?.hold_s ?? 0;
  const perPass = plan.steps.reduce((sum, step) => sum + 2 * step.hold_s, 0);
  return initial + plan.repeat * perPass;
}

/// A control file the runtime will ignore is worth refusing before the first write: the runtime ignores the whole
/// file until it changes again when a non-comment line is not NAME=value, so every marker wait would time out.
function checkControlText(text) {
  const lines = String(text).split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const body = lines[i].replace(/#.*$/, '').trim();
    if (!body) continue;
    if (!parseAssignment(lines[i])) {
      throw new PlanError(
        `the control file line ${i + 1} is not NAME=value; the runtime ignores the whole file and would never ` +
          'log a marker',
      );
    }
  }
}

/// Each value as the runtime holds it.
function normalized(map) {
  return Object.fromEntries(Object.entries(map).map(([name, value]) => [name, normalizeLiveValue(name, value)]));
}

/// The stats object's `live` map, every value as the runtime holds it, the HUD switch (left out while off) as off.
function liveMap(raw) {
  const out = {};
  for (const [name, value] of Object.entries(raw)) {
    const text = String(value);
    out[name] = isLiveName(name) ? (normalizeLiveValue(name, text) ?? text) : text;
  }
  if (!Object.hasOwn(out, UNLISTED_WHEN_OFF)) out[UNLISTED_WHEN_OFF] = 'off';
  return out;
}

/// One stats object as a sample, in either shape the runtime has written (stats-object.mjs): its seq (null for the
/// older shape), frame, the frames it covers (null for the older shape), frame_ms, its frames' times (null for the
/// older shape) and the live map; null for anything less (no frame_ms or no live map).
export function parseStats(text) {
  const record = parseStatsText(text);
  if (!record || !(record.frameMs > 0) || !record.live) return null;
  return {
    shape: record.shape,
    seq: record.seq,
    unixMs: record.unixMs,
    frame: record.frame,
    frames: record.interval?.frames ?? null,
    frame_ms: record.frameMs,
    list: record.frameMsList,
    fps: record.fps,
    live: liveMap(record.live),
    // The segments closed so far (`segments_closed_total`; `segment` in the pre-rebuild sample).
    segments_closed: record.segmentsClosed,
  };
}

/// A stored sample as the record objectsFigures joins.
const asRecord = (sample) => ({
  shape: sample.shape ?? 'sample',
  frameMs: sample.frame_ms,
  interval: sample.frames ? { frames: sample.frames } : null,
  frameMsList: sample.list ?? null,
});

/// The names in `live` whose value is not the one `wanted` gives.
function differences(live, wanted) {
  return Object.entries(wanted)
    .filter(([name, value]) => (live?.[name] ?? null) !== value)
    .map(([name, value]) => ({ name, want: value, seen: live?.[name] ?? '(absent)' }));
}

const differencesText = (list) =>
  list.map((entry) => `${entry.name}=${valueShown(entry.seen)} (asked ${valueShown(entry.want)})`).join(', ');

/// The stats file, read whole on every look. `look` says whether the object is the one read last time (the same text,
/// or the same `seq`), a new sample (`new`, with `missed`: objects written and never read, by the seq's jump), a
/// new text whose frame did not advance (`stalled`: the older shape only; the interval shape writes no object while
/// nothing presents), an object older than the last (`stale`: a lower seq, a new run), or a read that failed or did
/// not parse.
class StatsReader {
  constructor(file) {
    this.file = file;
    this.text = null;
    this.latest = null;
  }

  look() {
    let text;
    try {
      text = fs.readFileSync(this.file, 'utf8');
    } catch (err) {
      return { status: 'unparsable', why: err.code || 'unreadable' };
    }
    if (text === this.text) return { status: 'same', sample: this.latest };
    this.text = text;
    const sample = parseStats(text);
    if (!sample) return { status: 'unparsable', why: 'not a whole stats object' };
    const previous = this.latest;
    const order = sequenceOf(previous, sample);
    if (order?.kind === 'repeat') return { status: 'same', sample: this.latest };
    if (order?.kind === 'older') return { status: 'stale', sample };
    this.latest = sample;
    if (!order && previous && sample.frame <= previous.frame) return { status: 'stalled', sample };
    return { status: 'new', sample, missed: order?.missed ?? 0 };
  }
}

/// The stats file before the plan: a whole object with a live map, written within STATS_STALE_MS by its own clock
/// (`unix_ms`), or by the file's time for the older shape, which has none. Throws PlanError.
function readStart(file, staleMs, now = Date.now) {
  let stat;
  let text;
  try {
    stat = fs.statSync(file);
    text = fs.readFileSync(file, 'utf8');
  } catch (err) {
    throw new PlanError(
      `cannot read the stats file (${err.code || err.message}); the runner needs the switches in force before it ` +
        'writes any (a --live run writes stats.json beside live.txt)',
    );
  }
  const sample = parseStats(text);
  if (!sample) {
    throw new PlanError(
      'the stats file has no whole object with frame, frame_ms and a live map; the runner cannot know the start ' +
        'values, so it does not start',
    );
  }
  // By its own clock on the runner's clock (a test's virtual one); a file time only against the real one.
  const own = objectAgeMs(sample, now());
  const age = own ?? Date.now() - stat.mtimeMs;
  if (age > staleMs) {
    throw new PlanError(
      `the stats file was last written ${seconds(Math.round(age / 100) * 100)} ago ${own == null ? 'by its file time' : 'by its own clock'} ` +
        `(over ${seconds(staleMs)}): the game is not writing it, so its live map is not known to be what is in force`,
    );
  }
  return sample;
}

/// The live switch a plan's start writes (live.rs `Kind::Mark`). A plan may not set it: it is the runner's.
const MARK = 'LIVE_MARK';

/// A mark the runtime has not seen: one more than the larger of the one in force and the control file's.
function nextMark(inForce, controlText) {
  const fileValue = (controlText ?? '')
    .split(/\r?\n/)
    .map(parseAssignment)
    .find((assignment) => assignment?.name === MARK)?.value;
  const numbers = [inForce, fileValue].map((value) => Number(normalizeLiveValue(MARK, value ?? ''))).filter(Number.isFinite);
  return String(Math.max(0, ...numbers) + 1);
}

/// Writes a state through live.mjs's atomic merge: every name it holds, by value. Nothing is ever unset, since the
/// runtime keeps a removed name's value.
function writeControlState(file, state) {
  writeAtomic(file, mergeControl(readControl(file) ?? '', { set: state }));
}

/// The tail of a run log, starting where the runner started. It keeps only the parsed markers and segments, so a
/// long run's frame lines do not accumulate, and it notices the two failure strings on every line.
class LogWatcher {
  constructor(file, { now }) {
    this.file = file;
    this.now = now;
    const stat = fs.statSync(file);
    this.offset = stat.size;
    this.tail = '';
    this.seq = 0;
    this.markers = [];
    this.segments = [];
    this.pending = [];
    this.open = null;
    this.lastGrowthAt = now();
    this.failure = null;
    // A driver can be pointed at a title that already lost its device; check the log's recent tail once, without
    // keeping any of it, so the run stops before it measures a dead game.
    const tailStart = Math.max(0, stat.size - 64 * 1024);
    if (stat.size > tailStart) {
      const buffer = Buffer.alloc(stat.size - tailStart);
      const fd = fs.openSync(file, 'r');
      try {
        fs.readSync(fd, buffer, 0, buffer.length, tailStart);
      } finally {
        fs.closeSync(fd);
      }
      const failed = FAILURE_RE.exec(stripAnsi(buffer.toString('utf8')));
      if (failed) this.failure = /DXGI_ERROR/i.test(failed[0]) ? 'DXGI_ERROR' : 'device removed';
    }
  }

  poll() {
    let stat;
    try {
      stat = fs.statSync(this.file);
    } catch (err) {
      throw new StopError(`cannot read the run log (${err.code || err.message})`);
    }
    if (stat.size < this.offset) {
      this.offset = 0;
      this.tail = '';
    }
    if (stat.size === this.offset) return;
    const length = stat.size - this.offset;
    const buffer = Buffer.alloc(length);
    const fd = fs.openSync(this.file, 'r');
    try {
      fs.readSync(fd, buffer, 0, length, this.offset);
    } finally {
      fs.closeSync(fd);
    }
    this.offset = stat.size;
    this.lastGrowthAt = this.now();
    const text = this.tail + buffer.toString('utf8');
    const lines = text.split('\n');
    this.tail = lines.pop();
    for (const line of lines) this.consume(line);
  }

  consume(raw) {
    this.seq += 1;
    const text = stripAnsi(raw).replace(/\r$/, '');
    const failed = this.failure ? null : FAILURE_RE.exec(text);
    if (failed) this.failure = /DXGI_ERROR/i.test(failed[0]) ? 'DXGI_ERROR' : 'device removed';
    const parsed = parseLiveLine(raw);
    if (!parsed) return;
    // The open line is the segment under way, not one a change closed: kept apart, for the stop rule alone.
    if (parsed.openSegment) {
      this.open = { seq: this.seq, ...parsed.openSegment };
      return;
    }
    if (parsed.change) {
      const marker = { seq: this.seq, ...parsed.change };
      this.markers.push(marker);
      this.pending.push(parsed.change);
    } else {
      const segment = { seq: this.seq, ...parsed.segment, changes: this.pending };
      this.pending = [];
      this.segments.push(segment);
    }
  }

  markerAfter(seq, name) {
    return this.markers.find((marker) => marker.seq > seq && marker.name === name) ?? null;
  }

  markersAfter(seq, names) {
    return names.map((name) => this.markerAfter(seq, name));
  }

  /// The last segment line before `seq` and after `minSeq`: the stretch a change just closed.
  latestSegmentBefore(seq, minSeq = -1) {
    for (let i = this.segments.length - 1; i >= 0; i -= 1) {
      const segment = this.segments[i];
      if (segment.seq < seq && segment.seq > minSeq) return segment;
    }
    return null;
  }
}

/// The frames the segment under way must have counted before the five-times rule holds its mean against the baseline's.
export const OPEN_MIN_COUNTED = 10;

function safetyCheck({ watcher, safety, stallMs, now }) {
  if (watcher.failure) throw new StopError(`the log carries '${watcher.failure}'`);
  if (stallMs != null && now() - watcher.lastGrowthAt > stallMs) {
    throw new StopError(`the run log has not grown for ${seconds(stallMs)}; the game ended or hung`);
  }
  const overrun = (segment, what) => {
    if (segment.mean != null && safety.baselineMean != null && segment.mean > safety.baselineMean * 5) {
      throw new StopError(
        `${what} ${segment.segment} means ${one(segment.mean)} ms, over five times the first baseline's ` +
          `${one(safety.baselineMean)} ms; something broke`,
      );
    }
  };
  for (; safety.checked < watcher.segments.length; safety.checked += 1) {
    const segment = watcher.segments[safety.checked];
    if (segment.seq <= safety.armedSeq) continue;
    // A segment with nothing counted (all transition, light or long) has no mean: it is no reference.
    if (safety.baselineMean == null) safety.baselineMean = segment.mean;
    else overrun(segment, 'segment');
  }
  const open = watcher.open;
  if (open && open.seq > safety.armedSeq && open.counted >= OPEN_MIN_COUNTED) overrun(open, 'the open segment');
}

function checkAbort(signal) {
  if (signal?.aborted) throw new StopError('interrupted', 130);
}

/// Waits for a marker for every changed name, after `mark`. Returns the markers in `names` order.
async function waitForMarkers(ctx, names, mark) {
  if (!names.length) return [];
  const deadline = ctx.now() + ctx.markerTimeoutMs;
  for (;;) {
    checkAbort(ctx.signal);
    ctx.watcher.poll();
    safetyCheck(ctx);
    const markers = ctx.watcher.markersAfter(mark, names);
    if (markers.every(Boolean)) return markers;
    if (ctx.now() >= deadline) {
      const missing = names.filter((_, index) => !markers[index]);
      throw new StopError(
        `timed out after ${seconds(ctx.markerTimeoutMs)} waiting for the game to log ` +
          `${missing.join(', ')}; the game is not reading the control file`,
      );
    }
    await ctx.sleep(Math.min(ctx.pollMs, Math.max(1, deadline - ctx.now())));
  }
}

/// Waits for a stats sample newer than `afterFrame` whose live map holds every value of `state`, and returns it.
/// `guard` keeps the stop rules running (off for the last restore, which runs after a stop).
async function confirmState(ctx, state, afterFrame, what, { guard = true } = {}) {
  const deadline = ctx.now() + ctx.confirmTimeoutMs;
  for (;;) {
    if (guard) {
      checkAbort(ctx.signal);
      ctx.watcher.poll();
      safetyCheck(ctx);
    }
    ctx.stats.look();
    const sample = ctx.stats.latest;
    const fresh = sample && sample.frame > afterFrame;
    if (fresh && !differences(sample.live, state).length) return sample;
    if (ctx.now() >= deadline) {
      if (fresh) {
        throw new StopError(
          `${what}: ${seconds(ctx.confirmTimeoutMs)} after the write the stats file shows ` +
            `${differencesText(differences(sample.live, state))}; the game did not apply what was asked`,
        );
      }
      throw new StopError(
        `${what}: no new stats sample within ${seconds(ctx.confirmTimeoutMs)} of the write, so the switches in ` +
          'force cannot be confirmed',
      );
    }
    await ctx.sleep(Math.min(ctx.pollMs, Math.max(1, deadline - ctx.now())));
  }
}

/// Writes a state, waits for the markers of the names it changes, then for the stats file to show it. `liveMark`, at
/// the plan's start, is written beside the state as `LIVE_MARK` (live.rs: a new number changes no switch and ends the
/// segment under way), so the stretch before the plan, its loading with it, is a segment of its own; its marker is
/// waited for with the others.
async function applyState(ctx, state, what, { liveMark = null } = {}) {
  ctx.watcher.poll();
  safetyCheck(ctx);
  ctx.stats.look();
  const afterFrame = ctx.stats.latest?.frame ?? Number.NEGATIVE_INFINITY;
  const mark = ctx.watcher.seq;
  const changed = ctx.touched.filter((name) => state[name] !== ctx.current[name]);
  if (liveMark != null) changed.push(MARK);
  writeControlState(ctx.controlFile, liveMark == null ? state : { ...state, [MARK]: liveMark });
  ctx.wrote = true;
  const markers = await waitForMarkers(ctx, changed, mark);
  // With a mark, the confirming sample must show it too: one written before the mark applied holds the old segment count.
  const confirming = await confirmState(ctx, liveMark == null ? state : { ...state, [MARK]: liveMark }, afterFrame, what);
  ctx.current = { ...state };
  const firstSeq = markers.length ? Math.min(...markers.map((marker) => marker.seq)) : null;
  return { changed, mark, markers, firstSeq, confirming };
}

/// A new arm in the timeline: its target is every switch in force as the plan expects it (the start values with
/// the state on top), and its clock starts at the confirming sample, which is not one of its samples but gives the
/// closed-segment count every sample of the arm must keep. `settle_s` is its settling time.
function openArm(ctx, kind, label, state, confirming, settleS) {
  const arm = {
    index: ctx.timeline.arms.length,
    kind,
    label,
    target: { ...ctx.timeline.start, ...state, ...(ctx.timeline.mark == null ? {} : { [MARK]: ctx.timeline.mark }) },
    startFrame: confirming.frame,
    startSegment: confirming.segments_closed ?? null,
    settleMs: (settleS ?? 0) * 1000 * ctx.holdScale,
    startedMs: ctx.now() - ctx.planStart,
    endedMs: null,
    samples: [],
    excluded: { stalled: 0, stale: 0, unparsable: 0, missed: 0 },
  };
  ctx.timeline.arms.push(arm);
  return arm;
}

/// Holds `ms`, keeping the stop rules and recording every stats sample into `arm`.
async function measure(ctx, arm, ms) {
  const end = ctx.now() + ms;
  let lastNewAt = ctx.now();
  for (;;) {
    checkAbort(ctx.signal);
    ctx.watcher.poll();
    safetyCheck(ctx);
    const look = ctx.stats.look();
    const now = ctx.now();
    if (look.status === 'unparsable') {
      arm.excluded.unparsable += 1;
    } else if (look.status === 'stalled') {
      arm.excluded.stalled += 1;
      lastNewAt = now;
    } else if (look.status === 'stale') {
      arm.excluded.stale += 1;
    } else if (look.status === 'new') {
      const { shape, seq, frame, frames, frame_ms: frameMs, list, fps, live, segments_closed: closed } = look.sample;
      // at_ms: when the sample was read, from the arm's start (the settling rule reads it).
      const atMs = now - ctx.planStart - arm.startedMs;
      arm.samples.push({ shape, seq, frame, frames, frame_ms: frameMs, list, fps, live, segments_closed: closed, at_ms: atMs });
      arm.excluded.missed += look.missed;
      lastNewAt = now;
    } else if (now - lastNewAt > ctx.staleMs) {
      arm.excluded.stale += 1;
      lastNewAt = now;
    }
    arm.endedMs = now - ctx.planStart;
    const left = end - now;
    if (left <= 0) return;
    await ctx.sleep(Math.min(ctx.pollMs, left));
  }
}

function say(ctx, text) {
  ctx.out.write(`live-drive: ${text}\n`);
}

/// Plan `n`'s sidecar (1-based): beside the log, so `report --log` finds every plan's without another argument.
export function timelinePath(logFile, n = 1) {
  return n === 1 ? `${logFile}.live-drive.json` : `${logFile}.live-drive.${n}.json`;
}

/// Claims the next free sidecar beside the log by creating it exclusively, so two plans never share one. Returns
/// { n, file }.
function claimSidecar(logFile) {
  for (let n = 1; n < 10_000; n += 1) {
    const file = timelinePath(logFile, n);
    try {
      fs.writeFileSync(file, '', { flag: 'wx' });
      return { n, file };
    } catch (err) {
      if (err.code !== 'EEXIST') throw new PlanError(`cannot create the plan's sidecar beside the log (${err.code || err.message})`);
    }
  }
  throw new PlanError('every sidecar name beside the log is taken');
}

/// The lock and the state file beside a control file.
export const lockPath = (controlFile) => `${controlFile}.live-drive.lock`;
export const statePath = (controlFile) => `${controlFile}.live-drive-state.json`;

/// Whether a process with this id exists (signal 0 tests for it and delivers nothing).
export function processAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err.code === 'EPERM';
  }
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
}

/// Takes the control file's runner lock, or throws PlanError while a process holding it exists. A lock whose process
/// is gone (or that does not parse) is stale and replaced; `replaced` says so. Returns { body, replaced }.
function takeLock(file, { pid, alive }) {
  const body = JSON.stringify({ pid, startedAt: new Date().toISOString() });
  let replaced = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      fs.writeFileSync(file, body, { flag: 'wx' });
      return { body, replaced };
    } catch (err) {
      if (err.code !== 'EEXIST') {
        throw new PlanError(`cannot take the runner lock beside the control file (${err.code || err.message})`);
      }
    }
    const text = (() => {
      try {
        return fs.readFileSync(file, 'utf8');
      } catch {
        return null;
      }
    })();
    if (text === null) continue;
    let held = null;
    try {
      held = JSON.parse(text);
    } catch {
      held = null;
    }
    if (held && Number.isInteger(held.pid) && alive(held.pid)) {
      throw new PlanError(
        `another live-drive runner (process ${held.pid}, started ${held.startedAt ?? 'at an unknown time'}) holds ` +
          `${path.basename(file)}: one runner per control file. If no runner is running, delete that lock file`,
      );
    }
    replaced = held ?? { pid: null, startedAt: null };
    // Remove it only if it is still the stale one, then try again.
    try {
      if (fs.readFileSync(file, 'utf8') === text) fs.rmSync(file, { force: true });
    } catch {
      // Gone already.
    }
  }
  throw new PlanError(`cannot take the runner lock ${path.basename(file)}: another runner keeps taking it`);
}

/// Removes the lock if it is still this runner's.
function releaseLock(file, body) {
  try {
    if (fs.readFileSync(file, 'utf8') === body) fs.rmSync(file, { force: true });
  } catch {
    // Already gone.
  }
}

/// The state file an unfinished plan left, checked against the switches in force now. Returns a note when it is
/// cleared (the stats file shows its start values again); throws PlanError when some still differ.
function checkLeftover(file, live) {
  const left = readJson(file);
  if (!fs.existsSync(file)) return null;
  const who =
    `an unfinished plan (process ${left?.pid ?? 'unknown'}, started ${left?.startedAt ?? 'at an unknown time'}` +
    `${left?.sidecar ? `, sidecar ${path.basename(left.sidecar)}` : ''})`;
  const start = left?.start && typeof left.start === 'object' ? left.start : null;
  if (!start) {
    throw new PlanError(
      `${path.basename(file)} beside the control file is left by ${who} and does not hold its start values; check the ` +
        'switches in force by hand, then delete that file',
    );
  }
  const differ = Object.entries(start).filter(([name, value]) => (live[name] ?? '(absent)') !== value);
  if (differ.length) {
    throw new PlanError(
      `${who} was not finished and left switches away from its start values: ` +
        `${differ.map(([name, value]) => `${name} is ${valueShown(live[name] ?? '(absent)')}, was ${valueShown(value)}`).join('; ')}. ` +
        `Restore them (harness/live.sh set ${differ.map(([name, value]) => `${name}=${value}`).join(' ')}) and start again ` +
        `once the stats file shows them; if the game was restarted and these values are wanted, delete ${path.basename(file)}`,
    );
  }
  fs.rmSync(file, { force: true });
  return `${who} left ${path.basename(file)}; the stats file shows its start values again, so it is cleared`;
}

/// Runs a normalized plan against an open session. Returns { plan, occurrences, timeline }. Throws PlanError before
/// writing anything, and StopError (carrying `timeline`) after writing and confirming the baseline, or saying that
/// it could not.
export async function runPlan(input, options = {}) {
  const plan = normalizePlan(input);
  const opts = {
    logFile: options.logFile ?? null,
    controlFile: options.controlFile ?? null,
    statsFile: options.statsFile ?? null,
    markerTimeoutMs: options.markerTimeoutMs ?? MARKER_TIMEOUT_MS,
    confirmTimeoutMs: options.confirmTimeoutMs ?? CONFIRM_TIMEOUT_MS,
    staleMs: options.staleMs ?? STATS_STALE_MS,
    stallMs: options.stallMs ?? STALL_MS,
    pollMs: options.pollMs ?? POLL_MS,
    holdScale: options.holdScale ?? 1,
    sleep: options.sleep ?? realSleep,
    now: options.now ?? Date.now,
    signal: options.signal ?? null,
    out: options.out ?? process.stdout,
    err: options.err ?? process.stderr,
    writeTimeline: options.writeTimeline ?? true,
    picture: options.picture ?? true,
    pictureStepSeconds: options.pictureStepSeconds ?? 1,
    pictureMaxWidth: options.pictureMaxWidth ?? 960,
    // The process id the lock and state file record, and how a recorded one is told alive (tests replace both).
    pid: options.pid ?? process.pid,
    processAlive: options.processAlive ?? processAlive,
  };
  if (!opts.logFile) throw new PlanError('no run log: pass --log <path>');
  if (!opts.controlFile) throw new PlanError('no control file: pass --file <path> or set VICTORY_LIVE');
  if (!opts.statsFile) {
    throw new PlanError(
      'no stats file: pass --stats <path> or set VICTORY_LIVE_STATS (a --live run writes stats.json beside live.txt)',
    );
  }
  try {
    fs.accessSync(opts.logFile, fs.constants.R_OK);
  } catch (err) {
    throw new PlanError(`cannot read the run log (${err.code || err.message})`);
  }
  if (!(opts.holdScale > 0) || !Number.isFinite(opts.holdScale)) {
    throw new PlanError('holdScale must be a positive number');
  }
  const lockFile = lockPath(opts.controlFile);
  const lock = takeLock(lockFile, { pid: opts.pid, alive: opts.processAlive });
  try {
    return await runLocked(plan, opts, lock);
  } finally {
    releaseLock(lockFile, lock.body);
  }
}

async function runLocked(plan, opts, lock) {
  const controlText = readControl(opts.controlFile);
  if (controlText != null) checkControlText(controlText);

  const first = readStart(opts.statsFile, opts.staleMs, opts.now);
  const touched = LIVE_NAMES.filter(
    (name) => Object.hasOwn(plan.baseline, name) || plan.steps.some((step) => Object.hasOwn(step.set, name)),
  );
  const missing = touched.filter((name) => !Object.hasOwn(first.live, name));
  if (missing.length) {
    throw new PlanError(
      `the stats file's live map has no ${missing.join(', ')}: this build of the runtime does not have it, so its ` +
        'start value is unknown',
    );
  }
  const stateFile = statePath(opts.controlFile);
  const cleared = checkLeftover(stateFile, first.live);
  const start = first.live;
  const startTouched = Object.fromEntries(touched.map((name) => [name, start[name]]));
  const baselineState = { ...startTouched, ...normalized(plan.baseline) };
  const stateFor = (set) => ({ ...baselineState, ...normalized(set) });
  plan.steps.forEach((step, index) => {
    const state = stateFor(step.set);
    if (touched.every((name) => state[name] === baselineState[name])) {
      throw new PlanError(
        `step ${index + 1} ('${step.label}') sets ${setText(step.set)}, which the baseline already has ` +
          `(${Object.keys(step.set).some((name) => !Object.hasOwn(plan.baseline, name)) ? 'from the start values' : 'from the plan'}): ` +
          'its arms would compare the baseline with itself',
      );
    }
  });

  const watcher = new LogWatcher(opts.logFile, { now: opts.now });
  const stats = new StatsReader(opts.statsFile);
  stats.look();
  const safety = { baselineMean: null, checked: 0, armedSeq: Number.POSITIVE_INFINITY };
  const occurrences = occurrencesOf(plan);
  const timeline = {
    version: 2,
    plan,
    baseline: plan.baseline,
    touched,
    start,
    startFrame: first.frame,
    baselineState,
    // The plan's mark (`LIVE_MARK`), one more than the one in force; null where the runtime has no such switch.
    mark: Object.hasOwn(first.live, MARK) ? nextMark(first.live[MARK], controlText) : null,
    startedAt: new Date().toISOString(),
    finishedAt: null,
    restored: false,
    restoreNote: null,
    othersMoved: [],
    stopped: null,
    // Every `live:` change the log showed during the plan, with its frame (the report looks for one inside an arm).
    changes: watcher.markers,
    arms: [],
    occurrences: [],
  };
  const ctx = {
    ...opts,
    watcher,
    stats,
    safety,
    touched,
    current: { ...startTouched },
    planStart: opts.now(),
    wrote: false,
    timeline,
  };
  const sidecar = opts.writeTimeline ? claimSidecar(opts.logFile) : null;
  timeline.planNumber = sidecar?.n ?? null;
  let timelineWarned = false;
  const saveTimeline = () => {
    if (!sidecar) return;
    try {
      writeAtomic(sidecar.file, `${JSON.stringify(timeline, null, 2)}\n`);
    } catch (err) {
      if (!timelineWarned) {
        timelineWarned = true;
        opts.err.write(
          `live-drive: warning: cannot write the timeline beside the log (${err.code || err.message}); ` +
            'report will have no labels\n',
        );
      }
    }
  };
  if (lock.replaced) {
    say(ctx, `took over a stale runner lock (its process ${lock.replaced.pid ?? 'unknown'} no longer exists)`);
  }
  if (cleared) say(ctx, cleared);
  // The record a killed runner leaves: written before the first write, removed once the final baseline is confirmed.
  fs.writeFileSync(
    statePath(opts.controlFile),
    `${JSON.stringify(
      { version: 1, pid: opts.pid, startedAt: timeline.startedAt, log: opts.logFile, sidecar: sidecar?.file ?? null, start: startTouched },
      null,
      2,
    )}\n`,
  );
  saveTimeline();
  let failure = null;
  try {
    say(ctx, `start values (stats frame ${first.frame}): ${setText(startTouched) || '(none)'}`);
    say(ctx, `baseline: ${setText(baselineState) || '(none)'}`);
    if (sidecar && sidecar.n > 1) say(ctx, `this is plan ${sidecar.n} on this log; its sidecar is ${path.basename(sidecar.file)}`);
    const baselineWrite = await applyState(ctx, baselineState, 'the baseline', { liveMark: timeline.mark });
    if (timeline.mark != null) say(ctx, `marked the plan's start: ${MARK}=${timeline.mark} cuts the stretch before it into a segment of its own`);
    let baselineArm = openArm(ctx, 'baseline', null, baselineState, baselineWrite.confirming, plan.settle_s);
    const initialHoldMs = occurrences[0].hold_s * 1000 * opts.holdScale;
    say(ctx, `initial baseline hold: ${seconds(initialHoldMs)}`);
    saveTimeline();
    await measure(ctx, baselineArm, initialHoldMs);

    // Arm the segment checks after the baseline's markers (or its write, when it changed nothing). A pre-baseline
    // segment is before them; the A that closes at the first step is after them.
    const baselineSeq = baselineWrite.markers.length
      ? Math.max(...baselineWrite.markers.map((marker) => marker.seq))
      : baselineWrite.mark;
    safety.armedSeq = baselineSeq;
    safety.checked = watcher.segments.length;
    const associationMin = baselineSeq;
    saveTimeline();

    for (let i = 0; i < occurrences.length; i += 1) {
      const occurrence = occurrences[i];
      const record = {
        pass: occurrence.pass,
        direction: occurrence.direction,
        index: occurrence.index,
        label: occurrence.label,
        set: occurrence.set,
        hold_s: occurrence.hold_s,
        settle_s: occurrence.settle_s,
        beforeArm: baselineArm.index,
        stepArm: null,
        afterArm: null,
        beforeSegment: null,
        stepSegment: null,
        afterSegment: null,
      };
      timeline.occurrences.push(record);
      saveTimeline();

      const where = `${i + 1}/${occurrences.length} '${occurrence.label}'`;
      const stepState = stateFor(occurrence.set);
      say(ctx, `step ${where}: setting ${setText(occurrence.set)}`);
      const setResult = await applyState(ctx, stepState, `step ${where}`);
      if (setResult.firstSeq != null) {
        const closing = watcher.latestSegmentBefore(setResult.firstSeq, associationMin);
        if (closing) {
          record.beforeSegment = closing.segment;
          if (i > 0) timeline.occurrences[i - 1].afterSegment = closing.segment;
        }
      }
      const stepArm = openArm(ctx, 'step', occurrence.label, stepState, setResult.confirming, occurrence.settle_s);
      record.stepArm = stepArm.index;
      saveTimeline();

      const holdMs = occurrence.hold_s * 1000 * opts.holdScale;
      say(ctx, `step ${where}: confirmed at stats frame ${setResult.confirming.frame}; holding ${seconds(holdMs)}`);
      await measure(ctx, stepArm, holdMs);

      say(ctx, `step ${where}: restoring the baseline`);
      const restoreResult = await applyState(ctx, baselineState, `the baseline after step ${where}`);
      if (restoreResult.firstSeq != null) {
        const closing = watcher.latestSegmentBefore(restoreResult.firstSeq, associationMin);
        if (closing) record.stepSegment = closing.segment;
      }
      baselineArm = openArm(ctx, 'baseline', null, baselineState, restoreResult.confirming, occurrence.settle_s);
      record.afterArm = baselineArm.index;
      saveTimeline();

      say(ctx, `step ${where}: baseline confirmed; holding ${seconds(holdMs)}`);
      await measure(ctx, baselineArm, holdMs);
    }
    // The recording is fragmented, so it can be read while the title is still running. One pass covers every step
    // segment; a segment the log cannot place in time, a missing recording or a missing reference is said, not
    // failed. This runs after the last hold, so it cannot move a measured arm.
    if (opts.picture) {
      try {
        const note = await annotatePictureSegments({
          logFile: opts.logFile,
          segments: watcher.segments,
          occurrences: timeline.occurrences,
          stepSeconds: opts.pictureStepSeconds,
          maxWidth: opts.pictureMaxWidth,
        });
        if (note) say(ctx, `picture: ${note}`);
      } catch (err) {
        opts.err.write(`live-drive: warning: the picture check was skipped (${err.code || 'failed'})\n`);
      }
    }
    say(ctx, 'done');
    return { plan, occurrences: timeline.occurrences, timeline };
  } catch (err) {
    failure = err;
    throw err;
  } finally {
    if (!ctx.wrote) {
      timeline.restored = true;
      timeline.restoreNote = 'nothing was written';
    } else {
      try {
        stats.look();
        const afterFrame = stats.latest?.frame ?? Number.NEGATIVE_INFINITY;
        writeControlState(opts.controlFile, baselineState);
        if (failure) {
          say(ctx, `confirming the baseline (up to ${seconds(opts.confirmTimeoutMs)}; interrupt again to leave now)`);
        }
        const sample = await confirmState(ctx, baselineState, afterFrame, 'the baseline after the plan', {
          guard: false,
        });
        timeline.restored = true;
        timeline.restoreNote = `confirmed by the stats file at frame ${sample.frame}`;
        // The mark is the runner's own: it moved it.
        const others = Object.fromEntries(Object.entries(start).filter(([name]) => !touched.includes(name) && name !== MARK));
        timeline.othersMoved = differences(sample.live, others).map((entry) => ({
          name: entry.name,
          start: entry.want,
          now: entry.seen,
        }));
      } catch (err) {
        timeline.restored = false;
        timeline.restoreNote =
          err instanceof StopError ? err.message : `could not write the control file (${err.code || err.message})`;
      }
    }
    timeline.finishedAt = new Date().toISOString();
    // A confirmed baseline (or nothing written) ends the plan's record of unfinished business.
    if (timeline.restored) fs.rmSync(statePath(opts.controlFile), { force: true });
    if (failure) {
      timeline.stopped = failure.message;
      failure.timeline = timeline;
    }
    saveTimeline();
    if (timeline.restored) {
      say(ctx, ctx.wrote ? `baseline restored and ${timeline.restoreNote}` : 'nothing was written, nothing to restore');
      if (timeline.othersMoved.length) {
        say(
          ctx,
          `switches the plan does not touch moved during it: ${timeline.othersMoved
            .map((entry) => `${entry.name} ${valueShown(entry.start)} -> ${valueShown(entry.now)}`)
            .join(', ')}`,
        );
      }
    } else {
      opts.err.write(`live-drive: the baseline was written but is NOT confirmed: ${timeline.restoreNote}\n`);
      if (!failure) {
        // eslint-disable-next-line no-unsafe-finally
        throw Object.assign(new StopError(`the baseline is not confirmed after the plan: ${timeline.restoreNote}`), {
          timeline,
        });
      }
    }
  }
}

// ----------------------------------------------------------------------------------------------------------------
// report

/// Every plan's sidecar beside a log, in order: [{ n, file, timeline }], `timeline` null where one is not a sidecar
/// (not JSON, or an unknown version). Version 1 sidecars load too; the report calls every row of one unreliable.
export function loadTimelines(logFile) {
  const out = [];
  for (let n = 1; fs.existsSync(timelinePath(logFile, n)); n += 1) {
    const timeline = readJson(timelinePath(logFile, n));
    out.push({
      n,
      file: timelinePath(logFile, n),
      timeline: timeline && (timeline.version === 1 || timeline.version === 2) ? timeline : null,
    });
  }
  return out;
}

/// The newest plan's sidecar beside a log, or null when there is none.
export function loadTimeline(logFile) {
  return loadTimelines(logFile).at(-1)?.timeline ?? null;
}

/// Labels for a log without a sidecar: every segment a change cut. A restore also cuts a segment, so this can list
/// one row too many; the rows are unreliable anyway, since nothing confirmed what was in force.
function fallbackOccurrences(segments) {
  return segments
    .map((segment) => {
      if (!segment.changes.length) return null;
      const set = {};
      for (const change of segment.changes) set[change.name] = change.new;
      return {
        label: segment.changes.map((change) => `${change.name}=${change.new || '(empty)'}`).join(' '),
        set,
        pass: 1,
        direction: 'unknown',
      };
    })
    .filter(Boolean);
}

const LEGACY_REASON =
  'no stats samples: this run has an older sidecar or none, its runner restored the baseline by removing names ' +
  '(which the runtime ignores) and never confirmed what was in force, so its arms cannot be checked';

function emptyRow(label, set, reason) {
  return {
    label,
    set,
    inForce: {},
    median: null,
    mean: null,
    p95: null,
    baseline: null,
    difference: null,
    percent: null,
    spread: null,
    needed: null,
    neededPercent: null,
    passes: [],
    armCounts: [],
    verdict: 'unreliable',
    reason: [reason],
    excluded: { stalled: 0, stale: 0, unparsable: 0, missed: 0, settling: 0, pauses: 0 },
    repeats: [],
  };
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

const nearestRank = (sorted, p) =>
  sorted.length ? sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1))] : null;

/// One arm's figures (null figures with too few). Samples read in the arm's first settling time are left out. With
/// frame lists (the interval shape), a frame over PAUSE_FACTOR x the median of the arm's joined lists is left out as a
/// pause or hitch, and the median, mean and p95 are of the frames kept; without them (the older shape), a sample
/// whose frame_ms is over PAUSE_FACTOR x the median of the samples' is left out, the median is of the samples'
/// frame_ms, the mean their plain mean, and there is no p95. Over every sample read: the values in force by name, the
/// names whose samples disagree with the target or changed, a closed-segment count that moved from the confirming
/// sample's, and the log's changes inside the arm's frames.
function armFigures(arm, changes) {
  if (!arm) return null;
  const samples = Array.isArray(arm.samples) ? arm.samples : [];
  const settleMs = Number.isFinite(arm.settleMs) ? arm.settleMs : 0;
  const settling = samples.filter((sample) => Number.isFinite(sample.at_ms) && sample.at_ms < settleMs);
  const kept = samples.filter((sample) => !settling.includes(sample));
  const perFrame =
    kept.length > 0 &&
    kept.every((sample) => sample.shape === 'interval' && sample.frames > 0 && Array.isArray(sample.list) && sample.list.length > 0);
  const pauses = [];
  let good = 0;
  let frames = 0;
  let mean = null;
  let mid = null;
  let pauseLimit = null;
  const times = [];
  if (perFrame) {
    pauseLimit = PAUSE_FACTOR * median(kept.flatMap((sample) => sample.list));
    let wall = 0;
    for (const sample of kept) {
      const long = sample.list.filter((ms) => ms > pauseLimit);
      for (const ms of long) pauses.push({ ms, frame: sample.frame });
      const count = sample.frames - long.length;
      if (count > 0) good += 1;
      frames += Math.max(0, count);
      wall += sample.frame_ms * sample.frames - long.reduce((sum, ms) => sum + ms, 0);
      for (const ms of sample.list) if (ms <= pauseLimit) times.push(ms);
    }
    times.sort((a, b) => a - b);
    mean = frames > 0 ? wall / frames : null;
    mid = median(times);
  } else if (kept.length) {
    pauseLimit = PAUSE_FACTOR * median(kept.map((sample) => sample.frame_ms));
    const goodSamples = [];
    for (const sample of kept) {
      if (sample.frame_ms > pauseLimit) pauses.push({ ms: sample.frame_ms, frame: sample.frame });
      else goodSamples.push(sample);
    }
    good = goodSamples.length;
    const joined = objectsFigures(goodSamples.map(asRecord));
    const from = settling.length ? Math.max(...settling.map((sample) => sample.frame)) : arm.startFrame;
    const last = goodSamples.at(-1);
    frames = joined.frames ?? (last ? last.frame - from : 0);
    mean = joined.meanMs;
    mid = median(goodSamples.map((sample) => sample.frame_ms));
  }
  const tooFew = good < MIN_GOOD_SAMPLES || frames < MIN_ARM_FRAMES;
  const target = arm.target ?? {};
  const names = new Set([...Object.keys(target), ...samples.flatMap((sample) => Object.keys(sample.live ?? {}))]);
  const inForce = {};
  const wrong = [];
  const moved = [];
  for (const name of names) {
    const values = samples.map((sample) => sample.live?.[name] ?? '(absent)');
    const seen = [...new Set(values)];
    if (!seen.length) continue;
    inForce[name] = seen;
    const want = target[name] ?? '(absent)';
    const off = values.filter((value) => value !== want).length;
    if (seen.length > 1) moved.push({ name, seen });
    if (off) wrong.push({ name, want, seen, off });
  }
  // The closed-segment count: `segments_closed` (sidecar version 3), `segment` in a version-2 sidecar.
  const counts = samples.map((sample) => sample.segments_closed ?? sample.segment).filter(Number.isFinite);
  const segmentPath = [Number.isFinite(arm.startSegment) ? arm.startSegment : counts[0]];
  for (const count of counts) if (count !== segmentPath.at(-1)) segmentPath.push(count);
  const lastFrame = samples.length ? Math.max(...samples.map((sample) => sample.frame)) : null;
  const inside =
    lastFrame == null ? [] : changes.filter((change) => change.frame > arm.startFrame && change.frame <= lastFrame);
  return {
    arm: arm.index,
    kind: arm.kind,
    read: samples.length,
    good,
    frames,
    excluded: {
      stalled: 0,
      stale: 0,
      unparsable: 0,
      missed: 0,
      ...arm.excluded,
      settling: settling.length,
      pauses: pauses.length,
    },
    settleMs,
    pauses,
    pauseLimit,
    perFrame,
    times,
    shape: perFrame ? 'interval' : (kept[0]?.shape ?? null),
    median: tooFew ? null : mid,
    mean: tooFew ? null : mean,
    p95: tooFew || !perFrame ? null : nearestRank(times, 0.95),
    tooFew,
    inForce,
    wrong,
    moved,
    segmentMoved: segmentPath.length > 1 && segmentPath[0] != null ? segmentPath : null,
    inside,
    firstFrame: arm.startFrame,
    lastFrame,
  };
}

/// The step segment's per-frame p95 from the log, where that segment's in-force values agree with the step's.
function segmentP95(segment, target, touched) {
  if (!segment) return null;
  for (const name of touched) {
    if (!Object.hasOwn(segment.inForce, name)) continue;
    if ((normalizeLiveValue(name, segment.inForce[name]) ?? segment.inForce[name]) !== target[name]) return null;
  }
  return segment.p95;
}

const plural = (count, word) => `${count} ${word}${count === 1 ? '' : 's'}`;

function rowFor(label, resolved, timeline, bySegment, changes) {
  const arms = timeline.arms;
  const touched = timeline.touched ?? [];
  const reasons = [];
  const tooFew = [];
  const notes = [];
  const seenArms = new Map();
  const excluded = { stalled: 0, stale: 0, unparsable: 0, missed: 0, settling: 0, pauses: 0 };
  const figure = (index) => {
    if (index == null || !arms[index]) return null;
    if (!seenArms.has(index)) {
      const fig = armFigures(arms[index], changes);
      seenArms.set(index, fig);
      const name = `arm ${index} (${fig.kind === 'step' ? `step '${label}'` : 'baseline'})`;
      const frames = `frames ${fig.firstFrame}..${fig.lastFrame}`;
      for (const entry of fig.moved) {
        reasons.push(`${name}: ${entry.name} changed mid-arm (${entry.seen.map(valueShown).join(' -> ')}); set by hand?`);
      }
      for (const entry of fig.wrong) {
        if (fig.moved.some((moved) => moved.name === entry.name)) continue;
        reasons.push(
          `${name}: ${entry.name} was ${entry.seen.map(valueShown).join(', ')} in force, not the ` +
            `${valueShown(entry.want)} asked (${entry.off} of ${fig.read} samples)`,
        );
      }
      for (const change of fig.inside) {
        reasons.push(
          `${name}: the log shows ${change.name} ${valueShown(change.old)} -> ${valueShown(change.new)} at frame ` +
            `${change.frame}, inside the arm (${frames}), where the plan made no change`,
        );
      }
      if (fig.segmentMoved && !fig.inside.length) {
        reasons.push(
          `${name}: the stats file's count of closed segments went ${fig.segmentMoved.join(' -> ')} inside the arm ` +
            `(${frames}): a switch changed, and perhaps changed back, between two samples`,
        );
      }
      if (fig.tooFew) {
        tooFew.push(
          `${name}: too few samples (${fig.good} good over ${fig.frames} frames; ${MIN_GOOD_SAMPLES} over ` +
            `${MIN_ARM_FRAMES} needed)`,
        );
      }
      if (fig.excluded.settling) {
        notes.push(
          `${name}: ${plural(fig.excluded.settling, 'sample')} read in its first ${seconds(Math.round(fig.settleMs))} ` +
            'left out as settling (settle_s)',
        );
      }
      if (fig.pauses.length) {
        const shown = fig.pauses.slice(0, 3).map((pause) => `${one(pause.ms)} ms (stats frame ${pause.frame})`);
        if (fig.pauses.length > 3) shown.push(`${fig.pauses.length - 3} more`);
        notes.push(
          `${name}: ${plural(fig.pauses.length, fig.perFrame ? 'frame' : 'stats object')} over ${PAUSE_FACTOR} x the ` +
            `arm's median (${one(fig.pauseLimit / PAUSE_FACTOR)} ms) left out as a pause or hitch: ${shown.join(', ')}`,
        );
      }
      for (const key of Object.keys(excluded)) excluded[key] += fig.excluded[key] ?? 0;
    }
    return seenArms.get(index);
  };

  const passes = [];
  const baselines = [];
  const stepMedians = [];
  const armCounts = [];
  const repeats = [];
  const p95s = [];
  for (const occurrence of resolved) {
    const pass = `pass ${occurrence.pass}`;
    const before = figure(occurrence.beforeArm);
    const step = figure(occurrence.stepArm);
    const after = figure(occurrence.afterArm);
    if (!before) reasons.push(`${pass}: the baseline before the step was not measured`);
    if (!step) reasons.push(`${pass}: the step was not measured (the plan stopped first)`);
    if (step && !after) reasons.push(`${pass}: the baseline after the step was not measured (the plan stopped first)`);
    const baseline = before?.median != null && after?.median != null ? (before.median + after.median) / 2 : null;
    const effect = step?.median != null && baseline != null ? step.median - baseline : null;
    if (effect != null) {
      passes.push(effect);
      baselines.push(baseline);
      stepMedians.push(step.median);
    }
    armCounts.push(`${pass} ${[before, step, after].map((fig) => (fig ? `${fig.good} (${fig.frames})` : NA)).join('/')}`);
    const p95 = segmentP95(bySegment.get(Number(occurrence.stepSegment)), arms[occurrence.stepArm]?.target ?? {}, touched);
    if (p95 != null) p95s.push(p95);
    const summary = (fig) =>
      fig
        ? {
            arm: fig.arm,
            good: fig.good,
            frames: fig.frames,
            excluded: fig.excluded,
            median: round1(fig.median),
            mean: round1(fig.mean),
            inForce: Object.fromEntries(
              Object.entries(fig.inForce).filter(
                ([name]) => touched.includes(name) || fig.wrong.some((entry) => entry.name === name),
              ),
            ),
          }
        : null;
    repeats.push({
      pass: occurrence.pass,
      median: round1(step?.median),
      baseline: round1(baseline),
      difference: round1(effect),
      before: summary(before),
      step: summary(step),
      after: summary(after),
    });
  }

  const figures = [...seenArms.values()];
  const stepFigures = figures.filter((fig) => fig.kind === 'step');
  const baselineArmMedians = figures.filter((fig) => fig.kind === 'baseline' && fig.median != null).map((fig) => fig.median);
  const stepMedian = average(stepMedians);
  const baselineMedian = average(baselines);
  const difference = average(passes);
  const percent = difference != null && baselineMedian ? (difference / baselineMedian) * 100 : null;
  const spread = Math.max(range(passes), range(baselineArmMedians));
  const needed =
    baselineMedian != null ? Math.max((MIN_EFFECT_PERCENT / 100) * baselineMedian, SPREAD_FACTOR * spread) : null;
  const neededPercent = needed != null && baselineMedian ? (needed / baselineMedian) * 100 : null;

  let verdict;
  if (reasons.length) verdict = 'unreliable';
  else if (tooFew.length) verdict = 'too few samples';
  else if (difference == null || passes.length !== resolved.length) verdict = 'unreliable';
  else if (passes.length < 2) {
    verdict = ONE_PASS;
    notes.push(
      `one pass gives no direction: only a second pass measures the step arm's own noise (repeat ${DEFAULT_REPEAT} or more)`,
    );
  } else if (!passes.every((effect) => effect !== 0 && Math.sign(effect) === Math.sign(difference))) {
    verdict = 'no clear change';
    notes.push(`the passes disagree in sign (${passes.map((effect) => `${signed(round1(effect))} ms`).join(', ')})`);
  } else if (Math.abs(difference) <= needed) {
    verdict = 'no clear change';
    notes.push(
      `the ${one(Math.abs(difference))} ms difference is not over the ${one(needed)} ms needed (the larger of ` +
        `${MIN_EFFECT_PERCENT}% of the baseline and ${SPREAD_FACTOR} x the ${one(spread)} ms spread)`,
    );
  } else verdict = difference < 0 ? 'faster' : 'slower';

  const inForce = {};
  for (const name of Object.keys(resolved[0]?.set ?? {})) {
    const values = [...new Set(stepFigures.flatMap((fig) => fig.inForce[name] ?? []))];
    inForce[name] = values.length ? values : [];
  }

  const pictured = resolved.map((occurrence) => occurrence.picture).filter(Boolean);
  const picture = pictured.length
    ? pictured.reduce((worst, entry) =>
        pictureRank(entry.verdict) > pictureRank(worst.verdict) ? entry : worst,
      )
    : null;
  const rejected = [...new Set(resolved.flatMap((occurrence) => occurrence.rejected ?? []))];
  const tooFewFigures = verdict === 'too few samples';
  // The step arms' p95 from the frames they kept (the interval shape); the older shape has no lists, so the log's
  // step segments give it.
  const listP95 =
    stepFigures.length && stepFigures.every((fig) => fig.perFrame)
      ? nearestRank(stepFigures.flatMap((fig) => fig.times).sort((a, b) => a - b), 0.95)
      : null;
  const stepMeans = stepFigures.map((fig) => fig.mean).filter((value) => value != null);

  return {
    label,
    set: resolved[0]?.set ?? {},
    inForce,
    median: tooFewFigures ? null : round1(stepMedian),
    mean: tooFewFigures ? null : round1(average(stepMeans)),
    p95: tooFewFigures ? null : round1(listP95 ?? average(p95s)),
    p95Source: listP95 != null ? 'stats frame lists' : p95s.length ? 'log segments' : null,
    baseline: tooFewFigures ? null : round1(baselineMedian),
    difference: tooFewFigures ? null : round1(difference),
    percent: tooFewFigures || percent == null ? null : Math.round(percent * 10) / 10,
    spread: tooFewFigures ? null : round1(spread),
    needed: tooFewFigures ? null : round1(needed),
    neededPercent: tooFewFigures || neededPercent == null ? null : Math.round(neededPercent * 10) / 10,
    passes: passes.map(round1),
    armCounts,
    verdict,
    reason: [...reasons, ...tooFew, ...notes],
    excluded,
    ...(picture ? { picture } : {}),
    ...(rejected.length ? { rejected } : {}),
    repeats,
  };
}

/// Every `live:` change known for a plan: the ones its runner saw in the log, and the ones the log's segments carry.
function changesOf(segments, timeline) {
  const out = new Map();
  const listed = [...(Array.isArray(timeline.changes) ? timeline.changes : []), ...segments.flatMap((s) => s.changes ?? [])];
  for (const change of listed) {
    if (!change || !Number.isFinite(change.frame)) continue;
    out.set(`${change.name}|${change.frame}|${change.new}`, change);
  }
  return [...out.values()];
}

/// The report rows for parsed segments and one plan's sidecar (or null). Exported for the tests and for the command.
export function reportRows(segments, timeline) {
  if (!timeline || timeline.version !== 2 || !Array.isArray(timeline.arms)) {
    const occurrences = timeline?.occurrences?.length ? timeline.occurrences : fallbackOccurrences(segments);
    const rows = new Map();
    for (const occurrence of occurrences) {
      if (!rows.has(occurrence.label)) rows.set(occurrence.label, emptyRow(occurrence.label, occurrence.set ?? {}, LEGACY_REASON));
    }
    return [...rows.values()];
  }
  const bySegment = new Map(segments.map((segment) => [segment.segment, segment]));
  const changes = changesOf(segments, timeline);
  const groups = new Map();
  for (const occurrence of timeline.occurrences ?? []) {
    if (!groups.has(occurrence.label)) groups.set(occurrence.label, []);
    groups.get(occurrence.label).push(occurrence);
  }
  return [...groups.entries()].map(([label, resolved]) => rowFor(label, resolved, timeline, bySegment, changes));
}

function differenceText(row) {
  if (row.difference == null) return NA;
  const percent = row.percent == null ? '' : ` (${signed(row.percent)}%)`;
  return `${signed(row.difference)} ms${percent}`;
}

function neededText(row) {
  if (row.needed == null) return NA;
  return `${one(row.needed)} ms${row.neededPercent == null ? '' : ` (${one(row.neededPercent)}%)`}`;
}

function inForceText(row) {
  const parts = Object.entries(row.inForce ?? {}).map(
    ([name, values]) => `${name}=${values.length ? values.map(valueShown).join('|') : NA}`,
  );
  return parts.join(' ') || NA;
}

export function formatDriveReport(rows, timeline) {
  if (!rows.length) return 'no live steps';
  const out = [
    '# live-drive report',
    '| label | set to | in force (step arms) | step median (ms) | p95 (ms) | baseline median (ms) | difference | needed | passes (ms) | verdict |',
    '|---|---|---|---|---|---|---|---|---|---|',
  ];
  for (const row of rows) {
    const passes = row.passes.length ? row.passes.map(signed).join(', ') : NA;
    out.push(
      `| ${cell(row.label)} | ${cell(setText(row.set) || NA)} | ${cell(inForceText(row))} | ${one(row.median)} | ` +
        `${one(row.p95)} | ${one(row.baseline)} | ${differenceText(row)} | ${neededText(row)} | ${passes} | ${row.verdict} |`,
    );
  }
  const notes = rows.flatMap((row) => row.reason.map((reason) => `${row.label}: ${reason}`));
  for (const row of rows) {
    if (row.armCounts?.length) {
      notes.push(`${row.label}: good samples (frames) per arm, before/step/after: ${row.armCounts.join('; ')}`);
    }
    const { stalled = 0, stale = 0, unparsable = 0, missed = 0 } = row.excluded ?? {};
    if (stalled || stale || unparsable) {
      notes.push(
        `${row.label}: samples left out: ${stalled} whose frame did not advance, ${stale} stale ` +
          `(no rewrite for ${STATS_STALE_MS / 1000} s, or an older seq), ${unparsable} unparsable`,
      );
    }
    if (missed) notes.push(`${row.label}: ${missed} stats objects were written and never read (their seq jumped)`);
    if (row.p95Source) notes.push(`${row.label}: p95 from the ${row.p95Source}`);
  }
  if (notes.length) {
    out.push('', 'notes:');
    for (const note of notes) out.push(`- ${note}`);
  }
  const pictured = rows.filter((row) => row.picture);
  if (pictured.length) {
    out.push('', 'picture:');
    for (const row of pictured) {
      const rejected = row.rejected?.includes('picture') ? ' — rejected: picture' : '';
      const crops = Object.entries(row.picture.worstFrame?.crops ?? {})
        .map(([name, crop]) => `${name} ${crop.verdict}`)
        .join(', ');
      out.push(`- ${row.label}: ${row.picture.verdict}${rejected}${crops ? ` (${crops})` : ''}`);
    }
  }
  if (timeline?.stopped) out.push('', `runner stopped: ${timeline.stopped}`);
  if (timeline?.version === 2 && timeline.startedAt && !timeline.finishedAt) {
    out.push('', 'the plan has no end recorded: its runner was killed, or is still running');
  }
  if (timeline?.version === 2 && timeline.restored === false) {
    out.push('', `the baseline is NOT confirmed after the plan: ${timeline.restoreNote ?? 'no reason recorded'}`);
  }
  if (timeline?.othersMoved?.length) {
    out.push(
      '',
      `switches the plan does not touch moved during it: ${timeline.othersMoved
        .map((entry) => `${entry.name} ${valueShown(entry.start)} -> ${valueShown(entry.now)}`)
        .join(', ')}`,
    );
  }
  if (!timeline) {
    out.push('', 'no <log>.live-drive.json beside the log; labels come from the segment changes.');
  }
  out.push('', RULES_TEXT);
  return out.join('\n');
}

// ----------------------------------------------------------------------------------------------------------------
// command line

function usage(stream = process.stderr) {
  stream.write(
    'usage: live-drive.sh run <plan.json> --log <run log> [--file <control file>] [--stats <stats file>]\n' +
      '       live-drive.sh report --log <run log> [--plan <n>] [--json]\n' +
      'The control file comes from --file, else VICTORY_LIVE. The stats file comes from --stats, else\n' +
      'VICTORY_LIVE_STATS, else stats.json beside the control file. A plan is\n' +
      '  {"baseline": {NAME: value}, "steps": [{"label": "...", "set": {NAME: value}, "hold_s": 30, "settle_s": 0}],\n' +
      `   "repeat": ${DEFAULT_REPEAT}, "settle_s": ${DEFAULT_SETTLE_S}}\n` +
      `repeat defaults to ${DEFAULT_REPEAT} (one pass gives no verdict); settle_s (default ${DEFAULT_SETTLE_S}) is the seconds left out\n` +
      'as settling at the start of each arm. One runner per control file. Every plan on a log keeps its own\n' +
      "sidecar; report lists every plan, or the one --plan names. The rules are in live-drive.mjs's header.\n",
  );
}

function reportCommand(argv) {
  let logFile = null;
  let json = false;
  let only = null;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--log') {
      if (i + 1 >= argv.length) {
        console.error('live-drive: --log needs a path');
        return 2;
      }
      logFile = argv[(i += 1)];
    } else if (arg === '--plan') {
      only = Number(argv[(i += 1)]);
      if (!Number.isInteger(only) || only < 1) {
        console.error('live-drive: --plan needs a plan number (1 is the first plan on the log)');
        return 2;
      }
    } else if (arg === '--json') {
      json = true;
    } else if (arg === '--help' || arg === '-h') {
      usage(process.stdout);
      return 0;
    } else {
      console.error(`live-drive: unknown option or extra argument '${arg}' for report`);
      usage();
      return 2;
    }
  }
  if (!logFile) {
    console.error('live-drive: report needs --log <run log>');
    usage();
    return 2;
  }
  let text;
  try {
    text = fs.readFileSync(logFile, 'utf8');
  } catch (err) {
    console.error(`live-drive: cannot read the run log (${err.code || err.message})`);
    return 1;
  }
  const segments = parseLiveLog(text);
  const plans = loadTimelines(logFile);
  if (!plans.length) {
    if (only != null) {
      console.error(`live-drive: there is no plan ${only} beside this log (it has no sidecar)`);
      return 2;
    }
    const rows = reportRows(segments, null);
    if (json) console.log(JSON.stringify(rows, null, 2));
    else console.log(formatDriveReport(rows, null));
    return 0;
  }
  const chosen = only == null ? plans : plans.filter((entry) => entry.n === only);
  if (!chosen.length) {
    console.error(`live-drive: there is no plan ${only} beside this log (it has ${plans.length})`);
    return 2;
  }
  if (json) {
    const rows = chosen.flatMap((entry) =>
      entry.timeline ? reportRows(segments, entry.timeline).map((row) => ({ plan: entry.n, ...row })) : [],
    );
    console.log(JSON.stringify(rows, null, 2));
    return 0;
  }
  const blocks = chosen.map((entry) => {
    const started = entry.timeline?.startedAt ? `, started ${entry.timeline.startedAt}` : '';
    const heading =
      plans.length > 1 || only != null ? `## plan ${entry.n} of ${plans.length} (${path.basename(entry.file)}${started})\n` : '';
    if (!entry.timeline) return `${heading}its sidecar does not parse as a live-drive sidecar`;
    return heading + formatDriveReport(reportRows(segments, entry.timeline), entry.timeline);
  });
  console.log(blocks.join('\n\n'));
  return 0;
}

/// The stats file a run command uses: --stats, else VICTORY_LIVE_STATS, else stats.json beside the control file.
function statsFileFor(given, env, controlFile) {
  if (given) return given;
  if (env.VICTORY_LIVE_STATS) return env.VICTORY_LIVE_STATS;
  const beside = path.join(path.dirname(controlFile), 'stats.json');
  return fs.existsSync(beside) ? beside : null;
}

async function runCommand(argv, env) {
  let planFile = null;
  let logFile = null;
  let controlFile = null;
  let statsFile = null;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--log' || arg === '--file' || arg === '--stats') {
      if (i + 1 >= argv.length) {
        console.error(`live-drive: ${arg} needs a path`);
        return 2;
      }
      const value = argv[(i += 1)];
      if (arg === '--log') logFile = value;
      else if (arg === '--file') controlFile = value;
      else statsFile = value;
    } else if (arg === '--help' || arg === '-h') {
      usage(process.stdout);
      return 0;
    } else if (arg.startsWith('-')) {
      console.error(`live-drive: unknown option '${arg}'`);
      usage();
      return 2;
    } else if (planFile === null) {
      planFile = arg;
    } else {
      console.error(`live-drive: one plan at a time, not '${arg}'`);
      usage();
      return 2;
    }
  }
  if (!planFile) {
    console.error('live-drive: run needs a plan file');
    usage();
    return 2;
  }
  if (!logFile) {
    console.error('live-drive: run needs --log <run log>');
    usage();
    return 2;
  }
  const target = controlFile || env.VICTORY_LIVE;
  if (!target) {
    console.error('live-drive: no control file: pass --file <path> or set VICTORY_LIVE');
    return 2;
  }
  let input;
  try {
    input = JSON.parse(fs.readFileSync(planFile, 'utf8'));
  } catch (err) {
    console.error(`live-drive: cannot read the plan (${err.code || err.message})`);
    return 2;
  }
  let plan;
  try {
    plan = normalizePlan(input);
  } catch (err) {
    console.error(`live-drive: ${err.message}`);
    return err.exitCode || 2;
  }

  const controller = new AbortController();
  let signals = 0;
  const onSignal = () => {
    signals += 1;
    if (signals > 1) process.exit(130);
    controller.abort();
  };
  process.once('SIGINT', onSignal);
  process.once('SIGTERM', onSignal);
  try {
    await runPlan(plan, {
      logFile,
      controlFile: target,
      statsFile: statsFileFor(statsFile, env, target),
      signal: controller.signal,
    });
    return 0;
  } catch (err) {
    if (err instanceof StopError) {
      console.error(`live-drive: stopped: ${err.message}`);
      return err.exitCode;
    }
    console.error(`live-drive: ${err.message}`);
    return err.exitCode || 1;
  } finally {
    process.removeListener('SIGINT', onSignal);
    process.removeListener('SIGTERM', onSignal);
  }
}

/// The command line. Returns an exit code.
export async function main(argv, env = process.env) {
  if (!argv.length) {
    usage();
    return 2;
  }
  if (argv[0] === '--help' || argv[0] === '-h') {
    usage(process.stdout);
    return 0;
  }
  const [command, ...rest] = argv;
  if (command === 'run') return runCommand(rest, env);
  if (command === 'report') return reportCommand(rest);
  console.error(`live-drive: unknown command '${command}'`);
  usage();
  return 2;
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  main(process.argv.slice(2)).then(
    (code) => {
      process.exitCode = code;
    },
    (err) => {
      console.error(`live-drive: ${err.message}`);
      process.exitCode = 1;
    },
  );
}
