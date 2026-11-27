#!/usr/bin/env node
// The live switches' changes and the frame-time segments they cut, from a run log.
//
//   node harness/live-report.mjs <log> [--json]
//
// The runtime (victory-gpu `live.rs`) logs `live: <NAME> <old> -> <new> at frame <n>` for every applied change, a
// plan's mark among them (`LIVE_MARK`, a switch that changes nothing and only cuts a segment), and, when a change, a
// mark or a run end it sees closes a stretch, `live segment <k>: frames <a>..<b>, <n> counted, <n> excluded as
// transition (...; their mean <ms>), <n> excluded as light (...; their mean <ms>), <n> excluded as long (...; their mean
// <ms>), mean <ms>, p50 <ms>, p95 <ms> of the counted, begun at <...>, ended by <...>, skipped <d> draws and <s>
// dispatches, in force: <NAME=value ...>`; the segment under way logs the same with `open:` after each frame line
// (fixtures/runtime-lines.mjs holds each shape as the runtime's tests pin it). The pre-rebuild line (`frames <a>..<b>,
// mean <ms>, p50 <ms>, p95 <ms>, skipped ...`, every present counted) is read too, for old logs. A log line may carry
// ANSI colour codes, a CRLF ending and the timestamp, level and module before the text; all three are tolerated. A
// segment with fewer than 120 frames counted is marked `short`: its mean and percentiles are too few samples to trust.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { stripAnsi } from './parse-run.mjs';

export const SHORT_FRAMES = 120;
const DASH = '—';

// The segment's text, wherever it starts on the log line. `mean`, `p50` and `p95` are ms with one decimal.
// The pre-rebuild shape: every present counted, nothing left out.
const SEGMENT_OLD_RE =
  /live segment (\d+): frames (\d+)\.\.(\d+), mean ([\d.]+), p50 ([\d.]+), p95 ([\d.]+), skipped (\d+) draws and (\d+) dispatches, in force:\s*(.*)$/;
// The runtime's shape (live.rs `segment_line`): each frame in one of four, asked in this order: the first frames
// after a change (transition), a frame under the draws of a scene (light: a load, a menu or a pause), a frame over a
// second (long), or counted; the mean and percentiles are of the counted alone, and each kind left out has its mean.
const SEGMENT_BODY =
  String.raw`frames (\d+)\.\.(\d+), (\d+) counted, (\d+) excluded as transition \(the first (\d+) after a change; their mean ([\d.]+)\), ` +
  String.raw`(\d+) excluded as light \(under (\d+) draws: a load, a menu or a pause, the runtime cannot tell which; their mean ([\d.]+)\), ` +
  String.raw`(\d+) excluded as long \(over (\d+) ms; their mean ([\d.]+)\), mean ([\d.]+), p50 ([\d.]+), p95 ([\d.]+) of the counted, ` +
  String.raw`begun at (the start of the run|a change|a mark), `;
const SEGMENT_TAIL = String.raw`, skipped (\d+) draws and (\d+) dispatches, in force:\s*(.*)$`;
const SEGMENT_CLOSED_RE = new RegExp(
  String.raw`live segment (\d+): ${SEGMENT_BODY}ended by (a change|a mark|the end of the run)${SEGMENT_TAIL}`,
);
const SEGMENT_OPEN_RE = new RegExp(
  String.raw`live segment (\d+) open: ${SEGMENT_BODY}not ended \(the segment under way, as of this line\)${SEGMENT_TAIL}`,
);
// A marker's old or new value may be empty (an empty skip list), hence the non-greedy possibly-empty groups.
const CHANGE_RE = /live: ([A-Z0-9_]+) (.*?) -> (.*?) at frame (\d+)(?=\s|$)/;

/// `NAME=value` pairs as the segment line prints them, in the runtime's fixed order. An empty list stays an empty
/// string, which is what `NAME=` says.
export function parseInForce(text) {
  const values = {};
  for (const token of String(text).trim().split(/\s+/)) {
    if (!token) continue;
    const eq = token.indexOf('=');
    if (eq <= 0) continue;
    values[token.slice(0, eq)] = token.slice(eq + 1);
  }
  return values;
}

const BEGUN = { 'the start of the run': 'start', 'a change': 'change', 'a mark': 'mark' };
const ENDED = { 'a change': 'change', 'a mark': 'mark', 'the end of the run': 'end' };

/// A segment of the runtime's shape (closed or open) from its match. The open line has no `ended by` group.
function countedSegment(match, { open }) {
  const first = Number(match[2]);
  const last = Number(match[3]);
  const counted = Number(match[4]);
  // A segment with nothing counted prints 0.0 for its figures, and a kind with none left out 0.0 for its mean: they
  // are no figures.
  const figure = (text) => (counted ? Number(text) : null);
  const left = (count, mean) => (Number(count) ? Number(mean) : null);
  const tail = open ? 18 : 19;
  return {
    segment: Number(match[1]),
    shape: 'counted',
    open,
    frames: `${first}..${last}`,
    firstFrame: first,
    lastFrame: last,
    frameCount: last >= first ? last - first + 1 : 0,
    counted,
    transitionExcluded: Number(match[5]),
    transitionFrames: Number(match[6]),
    transitionMean: left(match[5], match[7]),
    lightExcluded: Number(match[8]),
    lightDraws: Number(match[9]),
    lightMean: left(match[8], match[10]),
    longExcluded: Number(match[11]),
    longMs: Number(match[12]),
    longMean: left(match[11], match[13]),
    mean: figure(match[14]),
    p50: figure(match[15]),
    p95: figure(match[16]),
    begun: BEGUN[match[17]],
    ended: open ? null : ENDED[match[18]],
    skippedDraws: Number(match[tail]),
    skippedDispatches: Number(match[tail + 1]),
    inForce: parseInForce(match[tail + 2]),
    short: counted < SHORT_FRAMES,
    meanChange: null,
    changes: [],
  };
}

/// One log line as a change, a closed segment or an open one (`openSegment`: the segment under way), or null for
/// anything else. Both segment shapes are read, and `shape` says which (`old`: the pre-rebuild line, every present
/// counted; `counted`: the runtime's, transition, light and long frames left out). The runner tails a log with this, so the line shapes live here once: colour codes
/// and a CRLF ending are tolerated, as in parseLiveLog.
export function parseLiveLine(raw) {
  const line = stripAnsi(raw).replace(/\r$/, '');
  const change = line.match(CHANGE_RE);
  if (change) {
    return {
      change: { name: change[1], old: change[2].trim(), new: change[3].trim(), frame: Number(change[4]) },
    };
  }
  const open = line.match(SEGMENT_OPEN_RE);
  if (open) return { openSegment: countedSegment(open, { open: true }) };
  const counted = line.match(SEGMENT_CLOSED_RE);
  if (counted) return { segment: countedSegment(counted, { open: false }) };
  const segment = line.match(SEGMENT_OLD_RE);
  if (!segment) return null;
  const first = Number(segment[2]);
  const last = Number(segment[3]);
  const count = last >= first ? last - first + 1 : 0;
  return {
    segment: {
      segment: Number(segment[1]),
      shape: 'old',
      open: false,
      frames: `${first}..${last}`,
      firstFrame: first,
      lastFrame: last,
      frameCount: count,
      counted: null,
      transitionExcluded: null,
      lightExcluded: null,
      longExcluded: null,
      begun: null,
      ended: null,
      mean: Number(segment[4]),
      p50: Number(segment[5]),
      p95: Number(segment[6]),
      skippedDraws: Number(segment[7]),
      skippedDispatches: Number(segment[8]),
      inForce: parseInForce(segment[9]),
      short: count < SHORT_FRAMES,
      meanChange: null,
      changes: [],
    },
  };
}

/// The segment still under way at the end of a log: the last open line whose number is past the last closed
/// segment's (a run ended by the harness's timer writes no closing line), or null.
export function lastOpenSegment(text) {
  let open = null;
  let closed = 0;
  for (const raw of String(text).split(/\r?\n/)) {
    const parsed = parseLiveLine(raw);
    if (parsed?.openSegment) open = parsed.openSegment;
    else if (parsed?.segment) closed = Math.max(closed, parsed.segment.segment);
  }
  return open && open.segment > closed ? open : null;
}

/// The segments in a run log, with the changes that cut each from the one before it. A change applies at the frame
/// boundary that closes the previous segment and is logged just after that segment's line, so it belongs to the
/// segment that follows. `meanChange` is against the previous segment and null for the first.
export function parseLiveLog(text) {
  const segments = [];
  let pending = [];
  for (const raw of String(text).split(/\r?\n/)) {
    const parsed = parseLiveLine(raw);
    if (!parsed) continue;
    if (parsed.change) {
      pending.push(parsed.change);
      continue;
    }
    // The open line is the segment under way, not a closed one: lastOpenSegment reads it.
    if (parsed.openSegment) continue;
    const segment = parsed.segment;
    segment.changes = pending;
    segments.push(segment);
    pending = [];
  }
  // Markers after the last segment (a log cut before the next one) belong to no segment shown.
  segments.forEach((segment, i) => {
    if (i === 0) return;
    if (segment.mean == null || segments[i - 1].mean == null) return;
    const delta = Math.round((segment.mean - segments[i - 1].mean) * 10) / 10;
    segment.meanChange = delta === 0 ? 0 : delta;
  });
  return segments;
}

const one = (value) => (Number.isFinite(value) ? value.toFixed(1) : 'n/a');
const cell = (value) => String(value).replace(/\|/g, '\\|');

function changedText(changes) {
  return changes
    .map((change) => `${change.name} ${change.old || '(empty)'} -> ${change.new || '(empty)'}`)
    .join('; ');
}

const BEGUN_TEXT = { start: 'the run start', change: 'a change', mark: 'a mark' };
const ENDED_TEXT = { change: 'a change', mark: 'a mark', end: 'the run end' };

/// What a segment's figures stand on: every present (the older shape), or the counted frames with the transition,
/// light and long frames left out, and how it began and ended.
function countedText(segment) {
  if (segment.shape !== 'counted') return 'every present (older line)';
  const how = `from ${BEGUN_TEXT[segment.begun]} to ${segment.open ? 'now (open)' : ENDED_TEXT[segment.ended]}`;
  return (
    `${segment.counted} counted; left out ${segment.transitionExcluded} transition, ${segment.lightExcluded} light, ` +
    `${segment.longExcluded} long; ${how}`
  );
}

/// The Markdown table: one row per segment, and the open one (`open`, the segment under way) last. The change
/// columns are dashes on the first row, because there is no segment before it to compare with.
export function formatReport(segments, { open = null } = {}) {
  if (!segments.length && !open) return 'no live segments';
  const out = [
    '# live segments',
    '| segment | frames | counted | mean | p50 | p95 | skipped draws | skipped dispatches | in force | mean change | changed |',
    '|---|---|---|---|---|---|---|---|---|---|---|',
  ];
  const rows = open ? [...segments, open] : segments;
  for (let i = 0; i < rows.length; i += 1) {
    const segment = rows[i];
    const frames = `${segment.frames}${segment.short ? ' (short)' : ''}`;
    const inForce = Object.entries(segment.inForce)
      .map(([name, value]) => `${name}=${value}`)
      .join(' ');
    const meanChange =
      i === 0 || segment.meanChange == null ? DASH : `${segment.meanChange > 0 ? '+' : ''}${one(segment.meanChange)} ms`;
    const changed = i === 0 ? DASH : segment.changes.length ? changedText(segment.changes) : DASH;
    out.push(
      `| ${segment.segment}${segment.open ? ' (open)' : ''} | ${cell(frames)} | ${cell(countedText(segment))} | ${one(segment.mean)} | ${one(segment.p50)} | ${one(segment.p95)} | ` +
        `${segment.skippedDraws} | ${segment.skippedDispatches} | ${cell(inForce || DASH)} | ${meanChange} | ${cell(changed)} |`,
    );
  }
  if (rows.some((segment) => segment.short)) {
    out.push('', `short: fewer than ${SHORT_FRAMES} frames counted; the mean and percentiles are too few samples to trust.`);
  }
  if (open) out.push('', `segment ${open.segment} is still open: its line is the newest as of the log's end (a run ended by a timer writes no closing line).`);
  return out.join('\n');
}

function usage() {
  console.error('usage: node live-report.mjs <log> [--json]');
}

/// Reads one log and prints its table (or, with `--json`, the same rows as objects). Returns an exit code.
export function main(argv) {
  let json = false;
  let log = null;
  for (const arg of argv) {
    if (arg === '--json') json = true;
    else if (arg === '--help' || arg === '-h') {
      usage();
      return 0;
    } else if (arg.startsWith('-')) {
      console.error(`live-report: unknown option ${arg}`);
      usage();
      return 2;
    } else if (log === null) log = arg;
    else {
      console.error(`live-report: one log at a time, not '${arg}'`);
      usage();
      return 2;
    }
  }
  if (log === null) {
    usage();
    return 2;
  }
  let text;
  try {
    text = fs.readFileSync(log, 'utf8');
  } catch (err) {
    console.error(`live-report: cannot read the log (${err.code || err.message})`);
    return 1;
  }
  const segments = parseLiveLog(text);
  const open = lastOpenSegment(text);
  if (json) console.log(JSON.stringify(open ? [...segments, open] : segments, null, 2));
  else console.log(formatReport(segments, { open }));
  return 0;
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) process.exit(main(process.argv.slice(2)));
