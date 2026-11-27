#!/usr/bin/env node
// Feeds the stream overlay's "live switches" panel: once a second it reads the running game's stats file (the
// runtime writes it when started with --live) and writes a small JSON file the overlay shows in a corner: every
// live switch and its value in force, the last few changes, and the frame time before and after the latest one.
// It only reads the run's files and writes the overlay's one; it cannot change a switch.
// Both stats shapes are read (stats-object.mjs): the interval shape's objects each cover every frame since the one
// before, so a repeated seq is skipped, a jump is counted as objects missed, and means are weighted by the frames each
// object covers; the older one-second sample has no seq, no median or worst frame (null, never zero) and no frame count.
//   node harness/live-overlay.mjs            # loop until stopped
//   node harness/live-overlay.mjs --once     # one pass (for a check)
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { objectAgeMs, readStatsObject, sequenceOf } from './stats-object.mjs';

const KEEP_CHANGES = 6;
const WINDOW = 20; // objects (about a second each) averaged on each side of a change
const ACTIVE_MS = 10_000; // an object older than this, by its own clock or else the file's, is no live run

/// The switches in a stats sample as a plain object, whatever shape the runtime wrote them in.
export function switchesOf(stats) {
  const live = stats && stats.live;
  if (!live) return {};
  if (Array.isArray(live)) {
    const out = {};
    for (const pair of live) {
      if (Array.isArray(pair) && pair.length >= 2) out[String(pair[0])] = String(pair[1]);
      else if (pair && typeof pair === 'object' && 'name' in pair) out[String(pair.name)] = String(pair.value);
    }
    return out;
  }
  if (typeof live === 'object') return Object.fromEntries(Object.entries(live).map(([k, v]) => [k, String(v)]));
  return {};
}

const round1 = (value) => (value == null ? null : Math.round(value * 10) / 10);

/// The mean frame time of `{ ms, frames }` entries: weighted by the frames each covers when every entry knows its
/// frames (the interval shape), else the plain mean of the entries (the older sample).
function meanOf(entries) {
  if (!entries.length) return null;
  if (entries.every((entry) => entry.frames > 0)) {
    const frames = entries.reduce((sum, entry) => sum + entry.frames, 0);
    return entries.reduce((sum, entry) => sum + entry.ms * entry.frames, 0) / frames;
  }
  return entries.reduce((sum, entry) => sum + entry.ms, 0) / entries.length;
}

const framesOf = (entries) =>
  entries.length && entries.every((entry) => entry.frames > 0) ? entries.reduce((sum, entry) => sum + entry.frames, 0) : null;

/// Whether `record` (readStatsObject) is a live run's: written less than 10 s before `now` by its own clock
/// (`unix_ms`), or, for the older shape without one (or a read caught mid-write, `record` null), by the file's
/// modification time `mtimeMs`.
export function isActive(record, mtimeMs, now = Date.now()) {
  const age = (record ? objectAgeMs(record, now) : null) ?? (Number.isFinite(mtimeMs) ? now - mtimeMs : null);
  return age !== null && age < ACTIVE_MS;
}

/// One step: the state after seeing `stats` (a parsed stats object of either shape) at `now` (milliseconds). `state`
/// starts as `{}`.
export function step(state, stats, now) {
  const record = readStatsObject(stats);
  const seen = sequenceOf(state.seq == null ? null : { seq: state.seq }, record);
  // A lower seq is the runtime writing this file afresh: the last run's changes and times are not this one's.
  const base = seen?.kind === 'older' ? {} : state;
  const next = {
    switches: base.switches || null,
    changes: base.changes ? base.changes.slice() : [],
    recent: base.recent ? base.recent.slice() : [],
    frame: base.frame ?? null,
    seq: base.seq ?? null,
    missed: base.missed ?? 0,
  };
  if (!record) return next;
  // The same object read again: by its seq, or, for the older shape without one, by its frame.
  if (seen ? seen.kind === 'repeat' : next.frame === record.frame) return next;
  if (seen) next.missed += seen.missed;
  next.frame = record.frame;
  next.seq = record.seq;
  const current = switchesOf(stats);
  if (next.switches) {
    for (const name of new Set([...Object.keys(next.switches), ...Object.keys(current)])) {
      const from = next.switches[name] ?? '(unset)';
      const to = current[name] ?? '(unset)';
      if (from === to) continue;
      next.changes.unshift({
        name, from, to, at: new Date(now).toISOString(), frame: record.frame,
        before_ms: round1(meanOf(next.recent)), after: [],
      });
    }
    next.changes = next.changes.slice(0, KEEP_CHANGES);
  }
  next.switches = current;
  if (record.frameMs !== null) {
    const entry = { ms: record.frameMs, frames: record.interval?.frames ?? null };
    next.recent.push(entry);
    if (next.recent.length > WINDOW) next.recent.shift();
    for (const change of next.changes) if (change.after.length < WINDOW) change.after.push(entry);
  }
  return next;
}

/// What the overlay reads. `stats_shape` says which shape the object was ('interval' or 'sample'); a figure the shape
/// lacks (the older sample's median and worst frame) is null, never zero. `frame_ms_worst` is the object's longest
/// frame (`frame_ms_worst`): an object of a second's few frames has no 95th percentile, and the runtime writes none.
export function panel(state, stats, now, active) {
  const record = readStatsObject(stats);
  return {
    updated: new Date(now).toISOString(),
    active,
    stats_shape: record?.shape ?? null,
    frame: record?.frame ?? null,
    fps: round1(record?.fps ?? null),
    frame_ms: round1(record?.frameMs ?? null),
    frame_ms_p50: round1(record?.frameMsP50 ?? null),
    frame_ms_worst: round1(record?.frameMsWorst ?? null),
    objects_missed: state.missed ?? 0,
    target_ms: 33.3,
    switches: Object.entries(state.switches || {}).map(([name, value]) => ({ name, value })),
    changes: (state.changes || []).map((c) => ({
      name: c.name, from: c.from, to: c.to, at: c.at, frame: c.frame,
      before_ms: c.before_ms, after_ms: round1(meanOf(c.after)), after_samples: c.after.length,
      after_frames: framesOf(c.after),
    })),
  };
}

function newestStats(captures) {
  let best = null;
  let entries = [];
  try { entries = fs.readdirSync(captures, { withFileTypes: true }); } catch { return null; }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const file = path.join(captures, entry.name, 'stats.json');
    try {
      const stat = fs.statSync(file);
      if (!best || stat.mtimeMs > best.mtimeMs) best = { file, mtimeMs: stat.mtimeMs };
    } catch { /* no stats file in this folder */ }
  }
  return best;
}

function writeAtomic(file, text) {
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, text);
  fs.renameSync(tmp, file);
}

function main(argv) {
  const once = argv.includes('--once');
  const captures = process.env.VICTORY_CAPTURES || 'victory-captures';
  const out = process.env.LIVE_OVERLAY_OUT || path.join(os.homedir(), '.config', 'victory-harness', 'stream', 'live.json');
  let state = {};
  let source = null;
  const tick = () => {
    const now = Date.now();
    const found = newestStats(captures);
    let stats = null;
    if (found) {
      if (found.file !== source) { source = found.file; state = {}; } // a new run: forget the last one's changes
      try { stats = JSON.parse(fs.readFileSync(found.file, 'utf8')); } catch { /* mid-write: next second */ }
    }
    const active = Boolean(found) && isActive(readStatsObject(stats), found.mtimeMs, now);
    if (active && stats) state = step(state, stats, now);
    try { writeAtomic(out, `${JSON.stringify(panel(state, active ? stats : null, now, active))}\n`); } catch { /* next second */ }
  };
  tick();
  if (!once) setInterval(tick, 1000);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main(process.argv.slice(2));
