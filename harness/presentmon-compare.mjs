#!/usr/bin/env node
// Checks the runtime's own frame times against PresentMon's for one run, over the same wall-clock window on both
// sides: the one count that does not come from the runtime's meter (the measurement audit of 2026-11-26; the check of
// the same day, jobs/2026-11-26-live-switches/check-tools-truth.md, found the two sides on different windows).
//
//   node harness/presentmon-compare.mjs <run log> [--csv <presentmon.csv>] [--stats <stats.jsonl>] [--pid N]
//        [--from-unix-ms A --to-unix-ms B | --from-seq S --to-seq T | --from-s A] [--to-s B] [--segment K]
//        [--tolerance-pct 5] [--runtime-mean MS] [--runtime-median MS] [--runtime-p95 MS]
//
// Inputs. The CSV defaults to <capture folder>/presentmon.csv (run-world.sh writes it when PRESENTMON_EXE is set), the
// stats history to <capture folder>/stats.jsonl (stats-recorder.mjs writes it on a --live run). PresentMon's rows are
// kept for the game's process alone, by the id the run recorded in its log (the priority, volume or screen-recording
// line), or --pid: rows are never pooled across processes, and a run with no recorded id is refused.
//
// The clocks. PresentMon's time column counts from its own start, and the watcher starts it at the heavy scene's first
// frame line, so that line's timestamp is the capture's zero (a delay in PresentMon's start after that line is not
// measured: a named limit). The runtime's frames are placed by the stats objects (`unix_ms` and `frame_ms_list`) when
// the run has a stats history, else by the frame lines' own timestamps (either shape).
//
// The window. Give it on the wall clock, as the settled stretch is known from the stats file: --from-unix-ms and
// --to-unix-ms, or --from-seq and --to-seq (two stats objects of the history: from the present before the first's
// frames to the second's last, `unix_ms` less `interval.wall_ms` to `unix_ms`). The tool does the arithmetic onto the
// capture's clock, prints the window on both sides with their frame counts, and refuses (COULD NOT COMPARE) when the
// capture or the runtime's frames do not cover it to within COVER_SLACK_MS; rows outside it count for nothing, so a
// capture of the whole run (presentmon.mjs, load-in included) is judged on the window alone. --from-s/--to-s (seconds
// on the capture's clock) are the older form: the window is then their overlap with the capture's rows and the
// runtime's frames, so it ends at the capture's last row at the latest, and with neither the whole capture is the
// window, load-in included. --segment K (closed, or open in a run the timer ended) takes the segment's counted frames
// instead, needs the capture to cover them, and is refused when the segment left light or long frames out of its
// figures (they are anywhere in it, and PresentMon's rows over its time hold them). Under MIN_OVERLAP_S seconds, or
// under MIN_FRAMES frames on either side, it is COULD NOT COMPARE.
//
// Each figure is the runtime's against PresentMon's over the window, as a share of PresentMon's (PresentMon is the
// judge, the owner, 2026-11-26), and agrees within the tolerance (5% by default):
// - frames: the runtime's frame counter over the window against PresentMon's rows (also within COUNT_SLACK_FRAMES).
// - mean: from the stats objects' frame lists; else the frame lines (the window clause's frames and wall ms, or an
//   older line's counter between two lines' timestamps); or the segment's.
// - median and p95: per-frame figures only: the stats objects' frame lists, the segment's counted frames, or given; a
//   frame line's own p50 and p95 only when the window is that one line's.
// Exit 0 PASS, 1 FAIL, 2 CANNOT CHECK (an input is missing: no CSV, no frame time or time column, no process id, no
// runtime frame) or a usage error, 3 COULD NOT COMPARE (the inputs are there, but they overlap too little).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { FRAME_RE, median, parseFrameWindow, percentile, readLogLines } from './parse-run.mjs';
import { lastOpenSegment, parseLiveLog } from './live-report.mjs';
import { loadLocalEnv } from './local-env.mjs';
import { captureFolderFromLog, csvNumber, presentMonCsvForLog, splitCsvLine } from './presentmon.mjs';
import { readStatsObject } from './stats-object.mjs';

export const DEFAULT_TOLERANCE_PCT = 5;
// Fewer frames than this on either side of the window is too few to say anything.
export const MIN_FRAMES = 30;
// A shorter overlap of the capture and the runtime's frames is too short to say anything.
export const MIN_OVERLAP_S = 10;
// The frame counts may differ by this many frames whatever the tolerance: a frame at each edge of the window.
export const COUNT_SLACK_FRAMES = 2;
// How far a segment's counted frames may run past the capture's first or last row (about one frame line's lag).
export const SEGMENT_SLACK_MS = 1000;
// How far a window given on the wall clock may run past the capture's or the runtime's first or last frame.
export const COVER_SLACK_MS = 1000;
export const EXIT = Object.freeze({ pass: 0, fail: 1, cannotCheck: 2, couldNotCompare: 3 });

// PresentMon's frame-time column under each version's name, and its time column with the unit it is in.
const FRAME_COLUMNS = ['frametime', 'msbetweenpresents'];
const TIME_COLUMNS = [
  ['timeinms', 1 / 1000],
  ['timeinseconds', 1],
  ['cpustarttimeinms', 1 / 1000],
  ['cpustarttime', 1 / 1000],
];
const PRESENTMON_STARTED_RE = /^presentmon: started at /;
// The lines a run writes that name the game's process id: run-world.sh's priority line (every run), set-volume.ps1's
// line (unless --volume off) and record-screen.mjs's (unless --no-record).
const PID_LINES = [
  [/\bpriority: .*? for pid ([\d, ]+) \(/, 'the priority line'],
  [/\bvolume: pid (\d+),/, 'the volume line'],
  [/\bwindow of pid (\d+)\b/, "the screen recorder's line"],
];

/// PresentMon's CSV text as { frameColumn, timeColumn, pidColumn, rows: [{ seconds, ms, pid }] }. A row without a frame
/// time is skipped; seconds is null for every row when the CSV has no time column, pid when it has no ProcessID.
export function readFrameTimes(text) {
  const lines = String(text).replace(/^﻿/, '').split(/\r?\n/);
  const headerAt = lines.findIndex((line) => line.trim() !== '');
  if (headerAt < 0) return { frameColumn: null, timeColumn: null, pidColumn: false, rows: [] };
  const names = splitCsvLine(lines[headerAt]).map((name) => name.trim().toLowerCase());
  const frameIndex = names.findIndex((name) => FRAME_COLUMNS.includes(name));
  const time = TIME_COLUMNS.map(([name, scale]) => ({ name, scale, index: names.indexOf(name) })).find((c) => c.index >= 0);
  const pidIndex = names.indexOf('processid');
  if (frameIndex < 0) return { frameColumn: null, timeColumn: time ? time.name : null, pidColumn: pidIndex >= 0, rows: [] };
  const rows = [];
  for (const line of lines.slice(headerAt + 1)) {
    if (!line.trim()) continue;
    const cells = splitCsvLine(line);
    const ms = csvNumber(cells[frameIndex]);
    if (ms === null) continue;
    const at = time ? csvNumber(cells[time.index]) : null;
    rows.push({ seconds: at === null ? null : at * time.scale, ms, pid: pidIndex >= 0 ? csvNumber(cells[pidIndex]) : null });
  }
  return { frameColumn: names[frameIndex], timeColumn: time ? time.name : null, pidColumn: pidIndex >= 0, rows };
}

/// Count, mean, median and nearest-rank p95 (parse-run.mjs's median and percentile, which presentmon.mjs uses too).
export function frameStats(values) {
  if (!values.length) return { count: 0, mean: null, median: null, p95: null };
  return {
    count: values.length,
    mean: values.reduce((sum, x) => sum + x, 0) / values.length,
    median: median(values),
    p95: percentile(values, 0.95),
  };
}

const stampMs = (line) => {
  const m = line.match(/^(\S+Z)\s/);
  const at = m ? Date.parse(m[1]) : NaN;
  return Number.isNaN(at) ? null : at;
};

/// The runtime's frame lines: { atMs, frame (null on builds that print none), ms, window (null on an older line) }.
export function runtimeFrames(lines) {
  const out = [];
  for (const line of lines) {
    const m = line.match(FRAME_RE);
    if (!m) continue;
    const atMs = stampMs(line);
    if (atMs === null) continue;
    out.push({
      atMs,
      frame: m.groups.frame === undefined ? null : Number(m.groups.frame),
      ms: Number(m.groups.ms),
      window: parseFrameWindow(line),
    });
  }
  return out;
}

/// The capture's zero on the log's clock: the last frame line before the watcher's "presentmon: started" line
/// (the line that triggered it). Null when the log has no such line.
export function captureZero(lines) {
  const started = lines.findIndex((line) => PRESENTMON_STARTED_RE.test(line));
  if (started < 0) return null;
  for (let i = started - 1; i >= 0; i -= 1) {
    if (FRAME_RE.test(lines[i])) return stampMs(lines[i]);
  }
  return null;
}

/// The live segments of a log, closed ones of either shape and the open one last (live-report.mjs reads them):
/// { number, first, last, counted, transitionExcluded, lightExcluded, longExcluded, mean, p50, p95, shape, open }.
export function liveSegments(lines) {
  const text = lines.join('\n');
  const open = lastOpenSegment(text);
  return [...parseLiveLog(text), ...(open ? [open] : [])].map((s) => ({
    number: s.segment,
    first: s.firstFrame,
    last: s.lastFrame,
    counted: s.counted,
    transitionExcluded: s.transitionExcluded,
    lightExcluded: s.lightExcluded,
    longExcluded: s.longExcluded,
    mean: s.mean,
    p50: s.p50,
    p95: s.p95,
    shape: s.shape,
    open: s.open,
  }));
}

/// The process ids the run's log names for the game, each with the lines that name it.
export function recordedPids(lines) {
  const found = new Map();
  for (const line of lines) {
    for (const [re, source] of PID_LINES) {
      const m = line.match(re);
      if (!m) continue;
      for (const id of m[1].split(/[,\s]+/).filter(Boolean).map(Number)) {
        if (!found.has(id)) found.set(id, []);
        if (!found.get(id).includes(source)) found.get(id).push(source);
      }
    }
  }
  return found;
}

/// The rows of the game's process: { pid, source, rows, others } or { error }.
function pickProcess(pm, lines, givenPid) {
  if (!pm.pidColumn) return { error: "the CSV has no ProcessID column, so its rows cannot be told to the game's process" };
  const candidates = givenPid === undefined ? recordedPids(lines) : new Map([[givenPid, ['--pid']]]);
  if (!candidates.size) {
    return {
      error:
        'the run recorded no process id for the game (no priority, volume or screen-recording line names one), and ' +
        "PresentMon's rows are never pooled across processes; give --pid N",
    };
  }
  const inCsv = new Set(pm.rows.map((row) => row.pid));
  const withRows = [...candidates.keys()].filter((id) => inCsv.has(id));
  if (!withRows.length) return { error: `no PresentMon row belongs to process ${[...candidates.keys()].join(', ')}` };
  if (withRows.length > 1) return { error: `processes ${withRows.join(', ')} all have rows in the CSV; give --pid N` };
  const pid = withRows[0];
  const rows = pm.rows.filter((row) => row.pid === pid);
  return { pid, source: candidates.get(pid).join(', '), rows, others: pm.rows.length - rows.length };
}

/// The runtime's frames from its stats objects (any order, repeats allowed): every listed frame as { n, at, ms }, `at`
/// its present on the wall clock (the object's `unix_ms` less the frames after it), in order. Objects of the older
/// sample shape carry no frame list and are left out.
export function statsFrames(objects) {
  const seen = new Set();
  const records = [];
  for (const raw of objects ?? []) {
    const record = readStatsObject(raw);
    if (!record || record.shape !== 'interval' || record.seq === null || seen.has(record.seq)) continue;
    const interval = record.interval;
    if (record.unixMs === null || !interval || interval.firstFrame === null || interval.wallMs === null) continue;
    if (!record.frameMsList?.length) continue;
    seen.add(record.seq);
    records.push(record);
  }
  records.sort((a, b) => a.seq - b.seq);
  const frames = [];
  let missing = 0;
  for (const record of records) {
    // The interval runs from the present before its first frame to its last frame's, `wall_ms` long; the list holds
    // its first frames in order (all of them unless `frame_ms_list_missing` says otherwise).
    let at = record.unixMs - record.interval.wallMs;
    record.frameMsList.forEach((ms, k) => {
      at += ms;
      frames.push({ n: record.interval.firstFrame + k, at, ms });
    });
    missing += record.frameMsListMissing ?? 0;
  }
  return { objects: records.length, records, frames, missing, byN: new Map(frames.map((f) => [f.n, f])) };
}

/// A frame number's time on the log's clock, by straight lines between the numbered frame lines around it.
function frameTime(frames, n) {
  const numbered = frames.filter((f) => f.frame !== null);
  for (let i = 1; i < numbered.length; i += 1) {
    const a = numbered[i - 1];
    const b = numbered[i];
    if (n >= a.frame && n <= b.frame && b.frame > a.frame) return a.atMs + ((n - a.frame) / (b.frame - a.frame)) * (b.atMs - a.atMs);
  }
  return null;
}

/// Frame n's present on the wall clock: { at, by }; { at, atLeast: true } for a frame after every frame known (a
/// segment closed at the run's end, past the last stats object and frame line); null when it cannot be placed.
function placeFrame(stats, frames, n) {
  const listed = stats.byN.get(n);
  if (listed) return { at: listed.at, by: "the stats objects' frame lists" };
  const next = stats.byN.get(n + 1);
  if (next) return { at: next.at - next.ms, by: "the stats objects' frame lists" };
  const between = frameTime(frames, n);
  if (between !== null) return { at: between, by: 'straight lines between the frame lines' };
  const known = [...frames.filter((f) => f.frame !== null).map((f) => ({ n: f.frame, at: f.atMs })), ...stats.frames];
  const latest = known.reduce((best, f) => (best === null || f.n > best.n ? f : best), null);
  return latest && n > latest.n ? { at: latest.at, atLeast: true, by: 'the last frame known' } : null;
}

/// The wall-clock span of frames a..b: from the present before a to b's, or null when they cannot be placed.
function placeFrames(stats, frames, a, b) {
  const from = placeFrame(stats, frames, a - 1);
  const to = placeFrame(stats, frames, b);
  if (!from || from.atLeast || !to) return null;
  return { from: from.at, to: to.at, toAtLeast: Boolean(to.atLeast), by: from.by === to.by ? from.by : `${from.by} and ${to.by}` };
}

/// Live segment K, closed (the last of that number) or the open one at the log's end.
function findSegment(lines, k) {
  const text = lines.join('\n');
  const closed = parseLiveLog(text).filter((s) => s.segment === k);
  if (closed.length) return closed.at(-1);
  const open = lastOpenSegment(text);
  return open && open.segment === k ? open : null;
}

const pct = (value, base) => ((value - base) / base) * 100;
const ms = (x) => (x === null || x === undefined ? 'n/a' : `${x.toFixed(1)} ms`);

/// The whole check, from the log's lines and the CSV's text; `options.statsObjects` is the stats history (parsed
/// objects). Returns { verdict: 'PASS' | 'FAIL' | 'CANNOT CHECK' | 'COULD NOT COMPARE', lines: [text], code }.
export function compareRun(logLines, csvText, options = {}) {
  const tolerance = options.tolerancePct ?? DEFAULT_TOLERANCE_PCT;
  const out = [];
  const refuse = (verdict, code, why) => ({ verdict, code, lines: [...out, `${verdict}: ${why}`] });
  const cannot = (why) => refuse('CANNOT CHECK', EXIT.cannotCheck, why);
  const couldNot = (why) => refuse('COULD NOT COMPARE', EXIT.couldNotCompare, why);
  if (csvText === null || csvText === undefined) return cannot('no PresentMon CSV for this run (set PRESENTMON_EXE in harness/local.env; --no-presentmon turns it off)');
  const pm = readFrameTimes(csvText);
  if (!pm.frameColumn) return cannot('the CSV has no frame time column (FrameTime or MsBetweenPresents)');
  if (!pm.timeColumn) return cannot("the CSV has no time column, so its rows cannot be placed on the run's clock");
  const zero = captureZero(logLines);
  if (zero === null) return cannot('the log has no "presentmon: started" line after a frame line, so the capture cannot be placed on its clock');
  const wallModes = [options.fromUnixMs, options.toUnixMs].filter((x) => x !== undefined).length;
  const seqModes = [options.fromSeq, options.toSeq].filter((x) => x !== undefined).length;
  if (wallModes === 1 || seqModes === 1) return cannot('a window on the wall clock needs both of its ends (--from-unix-ms with --to-unix-ms, --from-seq with --to-seq)');
  if ((wallModes || seqModes) && (wallModes && seqModes || options.fromS !== undefined || options.toS !== undefined || options.segment !== undefined)) {
    return cannot('give one window: --from-unix-ms/--to-unix-ms, --from-seq/--to-seq, --from-s/--to-s or --segment K');
  }
  const process = pickProcess(pm, logLines, options.pid);
  if (process.error) return cannot(process.error);
  const rows = process.rows.map((row) => ({ at: zero + row.seconds * 1000, ms: row.ms })).sort((a, b) => a.at - b.at);
  if (!rows.length) return couldNot(`no PresentMon row of process ${process.pid} has a frame time`);
  const capture = { from: rows[0].at, to: rows.at(-1).at };
  let asked = {
    from: zero + (options.fromS ?? 0) * 1000,
    to: options.toS === undefined ? Infinity : zero + options.toS * 1000,
  };
  const lines = runtimeFrames(logLines);
  const stats = statsFrames(options.statsObjects);
  const seconds = (wall) => ((wall - zero) / 1000).toFixed(1);
  out.push(`  the capture's zero: ${new Date(zero).toISOString()} (unix ms ${zero}), the last frame line before "presentmon: started"`);
  // A window given on the wall clock: placed on the capture's clock here, and covered by both sides or refused.
  let given = null;
  if (seqModes) {
    if (!stats.objects) return cannot('--from-seq/--to-seq need the run\'s stats history (--stats, or stats.jsonl in the capture folder)');
    const first = stats.records.find((record) => record.seq === options.fromSeq);
    const last = stats.records.find((record) => record.seq === options.toSeq);
    if (!first || !last) return cannot(`the stats history has no object of seq ${!first ? options.fromSeq : options.toSeq}`);
    if (last.seq < first.seq) return cannot('--to-seq is before --from-seq');
    given = { from: first.unixMs - first.interval.wallMs, to: last.unixMs, by: `stats objects seq ${first.seq}..${last.seq}` };
  } else if (wallModes) {
    if (!(options.toUnixMs > options.fromUnixMs)) return cannot('--to-unix-ms must be after --from-unix-ms');
    given = { from: options.fromUnixMs, to: options.toUnixMs, by: 'the wall clock given' };
  }
  if (given) {
    asked = given;
    const where = `the window ${given.from}..${given.to} unix ms (${given.by}; ${seconds(given.from)}-${seconds(given.to)} s on the capture's clock)`;
    if (capture.from > given.from + COVER_SLACK_MS || capture.to < given.to - COVER_SLACK_MS) {
      return couldNot(`the capture (${seconds(capture.from)}-${seconds(capture.to)} s) does not cover ${where}`);
    }
    const ours = stats.frames.length
      ? { from: stats.frames[0].at - stats.frames[0].ms, to: stats.frames.at(-1).at }
      : { from: lines.find((f) => f.frame !== null)?.atMs ?? Infinity, to: lines.filter((f) => f.frame !== null).at(-1)?.atMs ?? -Infinity };
    if (ours.from > given.from + COVER_SLACK_MS || ours.to < given.to - COVER_SLACK_MS) {
      return couldNot(`the runtime's frames (${seconds(ours.from)}-${seconds(ours.to)} s) do not cover ${where}`);
    }
    out.push(`  asked: ${where}`);
  }

  // The window and the runtime's figures for it, from the segment, the stats history or the frame lines.
  const runtime = { mean: null, median: null, p95: null, frames: null };
  const source = { mean: null, median: null, p95: null };
  let window;
  let runtimeText;
  const notes = [];
  if (options.segment !== undefined) {
    const segment = findSegment(logLines, options.segment);
    if (!segment) return cannot(`no live segment ${options.segment} in the log, closed or open`);
    const excluded = segment.shape === 'counted' ? segment.transitionExcluded : 0;
    const counted = segment.shape === 'counted' ? segment.counted : segment.frameCount;
    if (!counted || segment.mean === null) return cannot(`live segment ${options.segment} counted no frames, so it has no figures`);
    // The transition frames are the segment's first; light and long frames are anywhere in it, left out of its
    // figures, and PresentMon's rows over the same time hold them: the two sides would not stand on the same frames.
    if (segment.shape === 'counted' && (segment.lightExcluded || segment.longExcluded)) {
      return couldNot(
        `live segment ${options.segment} left ${segment.lightExcluded} light and ${segment.longExcluded} long frames out of its ` +
          "figures, anywhere in it, and PresentMon's rows over its time hold them; compare a window on the wall clock instead",
      );
    }
    const a = segment.firstFrame + excluded;
    const b = segment.lastFrame;
    const span = placeFrames(stats, lines, a, b);
    if (!span) return cannot(`live segment ${options.segment}'s frames ${a}..${b} cannot be placed on the clock`);
    window = span;
    if (span.from < capture.from - SEGMENT_SLACK_MS || span.to > capture.to + SEGMENT_SLACK_MS) {
      return couldNot(
        `the capture (${seconds(capture.from)}-${seconds(capture.to)} s) does not cover live segment ${options.segment}'s ` +
          `counted frames ${a}..${b} (${seconds(span.from)}-${span.toAtLeast ? 'past ' : ''}${seconds(span.to)} s)`,
      );
    }
    if (span.toAtLeast) return cannot(`live segment ${options.segment} ends past the last frame the log and the stats history place, so its end cannot be placed`);
    runtime.mean = segment.mean;
    runtime.median = segment.p50;
    runtime.p95 = segment.p95;
    runtime.frames = counted;
    const label = `live segment ${segment.segment}`;
    Object.assign(source, { mean: label, median: label, p95: label });
    const shape =
      segment.shape === 'counted'
        ? `the runtime's shape, ${segment.open ? "open (the segment under way at the log's end), " : ''}${counted} counted, ${excluded} transition frames left out, no light or long frame`
        : 'the pre-rebuild shape, every present counted';
    runtimeText = `${counted} frames by its frame counter, frames ${a}..${b} (${label}: ${shape}; placed by ${span.by})`;
  } else if (stats.frames.length) {
    const from = Math.max(asked.from, capture.from, stats.frames[0].at - stats.frames[0].ms);
    const to = Math.min(asked.to, capture.to, stats.frames.at(-1).at);
    window = { from, to };
    const inside = stats.frames.filter((f) => f.at > from && f.at <= to);
    if (inside.length) {
      const figures = frameStats(inside.map((f) => f.ms));
      runtime.mean = figures.mean;
      runtime.median = figures.median;
      runtime.p95 = figures.p95;
      runtime.frames = inside.at(-1).n - inside[0].n + 1;
      const label = `the stats objects' frame lists, frames ${inside[0].n}..${inside.at(-1).n}`;
      Object.assign(source, { mean: label, median: label, p95: label });
      if (runtime.frames !== inside.length) {
        notes.push(`  note: ${runtime.frames - inside.length} frames in the window are in no recorded stats object's list; the mean, median and p95 stand on ${inside.length}`);
      }
    }
    runtimeText = `${runtime.frames ?? 0} frames by its frame counter (the stats history: ${stats.objects} objects)`;
  } else {
    const from = Math.max(asked.from, capture.from);
    const to = Math.min(asked.to, capture.to);
    const inside = lines.filter((f) => f.frame !== null && f.atMs >= from && f.atMs <= to);
    if (inside.length < 2 || inside.at(-1).frame <= inside[0].frame) {
      return couldNot(`fewer than two numbered frame lines inside the capture's window (${seconds(from)}-${Number.isFinite(to) ? seconds(to) : 'end'} s)`);
    }
    const first = inside[0];
    const last = inside.at(-1);
    window = { from: first.atMs, to: last.atMs };
    runtime.frames = last.frame - first.frame;
    const windows = inside.slice(1).map((f) => f.window);
    if (windows.every(Boolean)) {
      // The window clause: each line's frames and wall time on the runtime's own clock, weighted by frames.
      const frames = windows.reduce((sum, w) => sum + w.frames, 0);
      const wall = windows.reduce((sum, w) => sum + w.wallMs, 0);
      runtime.mean = wall / frames;
      source.mean = `frames ${windows[0].firstFrame}..${windows.at(-1).lastFrame}: ${frames} frames in ${wall.toFixed(0)} ms by its frame lines' windows`;
      if (windows.length === 1) {
        runtime.median = windows[0].p50Ms;
        runtime.p95 = windows[0].p95Ms;
        source.median = source.p95 = "the frame line's window";
      } else {
        out.push(
          `runtime's frame lines' own p50 and p95 (one window each, which do not join; for reference): ` +
            `p50 median ${ms(median(windows.map((w) => w.p50Ms)))}, p95 largest ${ms(Math.max(...windows.map((w) => w.p95Ms)))}`,
        );
      }
      const gaps = windows.filter((w, i) => w.firstFrame !== (i === 0 ? first.frame : windows[i - 1].lastFrame) + 1).length;
      if (gaps) notes.push(`  note: ${gaps} frame lines' windows do not start where the line before ended (a line missed)`);
    } else {
      runtime.mean = (last.atMs - first.atMs) / runtime.frames;
      source.mean = `frames ${first.frame}..${last.frame} by its counter`;
      const lineMeans = frameStats(inside.slice(1).map((f) => f.ms));
      out.push(`runtime's frame lines in the window (older shape, 5-second means, for reference): median ${ms(lineMeans.median)}, p95 ${ms(lineMeans.p95)}`);
    }
    runtimeText = `${runtime.frames} frames by its frame counter, frames ${first.frame}..${last.frame} (the frame lines' timestamps; no stats history, so the median and p95 need --segment K)`;
  }
  for (const [key, given] of [['mean', options.runtimeMean], ['median', options.runtimeMedian], ['p95', options.runtimeP95]]) {
    if (given === undefined || given === null) continue;
    runtime[key] = given;
    source[key] = 'given';
  }

  const inWindow = rows.filter((row) => row.at > window.from && row.at <= window.to);
  const pmStats = frameStats(inWindow.map((row) => row.ms));
  const overlapS = Math.max(0, (window.to - window.from) / 1000);
  out.unshift(
    `presentmon-compare: window ${seconds(window.from)}-${seconds(window.to)} s of the capture (${overlapS.toFixed(1)} s; ` +
      `unix ms ${Math.round(window.from)}..${Math.round(window.to)}; the capture's rows ${seconds(capture.from)}-${seconds(capture.to)} s); ` +
      `tolerance ${tolerance}%`,
    `  PresentMon: ${pmStats.count} frames (${pm.frameColumn}) of process ${process.pid} (its id from ${process.source}` +
      `${process.others ? `; ${process.others} rows of other processes left out` : ''})`,
    `  runtime: ${runtimeText}`,
    "  the clocks are tied by the trigger frame line's time (PresentMon's zero); a delay in PresentMon's start after it is not measured",
    ...notes,
  );
  if (overlapS < MIN_OVERLAP_S) return couldNot(`the capture and the runtime's frames overlap ${overlapS.toFixed(1)} s, under ${MIN_OVERLAP_S} s`);
  if (pmStats.count < MIN_FRAMES) return couldNot(`${pmStats.count} PresentMon frames in the window, under ${MIN_FRAMES}`);
  if (runtime.frames === null || runtime.frames < MIN_FRAMES) return couldNot(`${runtime.frames ?? 0} runtime frames in the window, under ${MIN_FRAMES}`);

  const checked = [];
  const failed = [];
  for (const key of ['mean', 'median', 'p95']) {
    const theirs = pmStats[key];
    const ours = runtime[key];
    if (ours === null || ours === undefined || !(ours > 0)) {
      out.push(`  ${key.padEnd(6)} PresentMon ${ms(theirs)}; runtime n/a (no per-frame figure for this window): not checked`);
      continue;
    }
    const diff = pct(ours, theirs);
    const ok = Math.abs(diff) <= tolerance;
    checked.push(key);
    if (!ok) failed.push(key);
    out.push(
      `  ${key.padEnd(6)} runtime ${ms(ours)} (${source[key] ?? 'given'}) against PresentMon ${ms(theirs)}: ` +
        `${diff >= 0 ? '+' : ''}${diff.toFixed(1)}% ${ok ? 'agrees' : 'DISAGREES'}`,
    );
  }
  const countDiff = pct(runtime.frames, pmStats.count);
  const countOk = Math.abs(countDiff) <= tolerance || Math.abs(runtime.frames - pmStats.count) <= COUNT_SLACK_FRAMES;
  checked.push('frames');
  if (!countOk) failed.push('frames');
  out.push(
    `  ${'frames'.padEnd(6)} runtime ${runtime.frames} against PresentMon ${pmStats.count}: ` +
      `${countDiff >= 0 ? '+' : ''}${countDiff.toFixed(1)}% ${countOk ? 'agrees' : 'DISAGREES'}`,
  );
  if (failed.length) {
    return { verdict: 'FAIL', code: EXIT.fail, lines: [...out, `FAIL: the runtime differs from PresentMon by more than ${tolerance}% (${failed.join(', ')})`] };
  }
  return { verdict: 'PASS', code: EXIT.pass, lines: [...out, `PASS: the runtime agrees with PresentMon within ${tolerance}% (${checked.join(', ')})`] };
}

/// A stats history file (one JSON object a line, as stats-recorder.mjs writes it) as parsed objects; a line that does
/// not parse is skipped.
export function readStatsHistory(text) {
  const objects = [];
  for (const line of String(text).split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      objects.push(JSON.parse(line));
    } catch {
      // A line cut by the recorder's end: the objects before it stand.
    }
  }
  return objects;
}

function usage() {
  console.error(
    'usage: node harness/presentmon-compare.mjs <run log> [--csv <presentmon.csv>] [--stats <stats.jsonl>] [--pid N] ' +
      '[--from-unix-ms A --to-unix-ms B | --from-seq S --to-seq T | --from-s A --to-s B | --segment K] [--tolerance-pct 5] ' +
      '[--runtime-mean MS] [--runtime-median MS] [--runtime-p95 MS]',
  );
  return EXIT.cannotCheck;
}

export function main(argv) {
  const options = {};
  let log = null;
  let csv = null;
  let statsFile = null;
  const numberFlags = {
    '--from-s': 'fromS',
    '--from-unix-ms': 'fromUnixMs',
    '--to-unix-ms': 'toUnixMs',
    '--from-seq': 'fromSeq',
    '--to-seq': 'toSeq',
    '--to-s': 'toS',
    '--segment': 'segment',
    '--pid': 'pid',
    '--tolerance-pct': 'tolerancePct',
    '--runtime-mean': 'runtimeMean',
    '--runtime-median': 'runtimeMedian',
    '--runtime-p95': 'runtimeP95',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--csv') csv = argv[++i];
    else if (arg === '--stats') statsFile = argv[++i];
    else if (numberFlags[arg]) {
      const value = Number(argv[++i]);
      if (!Number.isFinite(value) || value < 0) return usage();
      options[numberFlags[arg]] = value;
    } else if (arg.startsWith('--') || log) return usage();
    else log = arg;
  }
  if (!log) return usage();
  loadLocalEnv();
  const lines = readLogLines(log);
  const csvPath = csv ?? presentMonCsvForLog(lines, log);
  const csvText = csvPath && fs.existsSync(csvPath) ? fs.readFileSync(csvPath, 'utf8') : null;
  const folder = captureFolderFromLog(lines, log);
  const statsPath = statsFile ?? (folder ? `${folder}/stats.jsonl` : null);
  if (statsPath && fs.existsSync(statsPath)) options.statsObjects = readStatsHistory(fs.readFileSync(statsPath, 'utf8'));
  const result = compareRun(lines, csvText, options);
  for (const line of result.lines) console.log(line);
  return result.code;
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) process.exit(main(process.argv.slice(2)));
