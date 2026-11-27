#!/usr/bin/env node
// Keeps every object of the runtime's live stats file. The runtime rewrites the file (VICTORY_LIVE_STATS, a --live
// run's stats.json) about once a second, each object covering every frame since the one before, so after the run only
// the last is left; this appends each new one to a history, one JSON object a line, for a tool that reads the run's
// stats over a window afterwards (presentmon-compare.mjs takes its per-frame median and p95 from the frame lists).
//
//   node harness/stats-recorder.mjs <stats.json> <stats.jsonl> --log <run log> [--poll-ms 250]
//
// An object is new when its seq differs from the last one appended (the older sample shape, which has no seq: when its
// frame differs); a seq that jumps is counted as objects missed. A file that does not parse (read mid-write, or between
// the runtime's writes) is read again at the next poll. It ends one last read after the log carries run-world.sh's
// "harness: load end" line, written once the game has exited, and says how many it kept.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseStatsText } from './stats-object.mjs';

export const DEFAULT_POLL_MS = 250;
// run-world.sh writes this once the game process has ended (after the end load sample).
export const LOAD_END_RE = /^harness: load end exit=/m;
const LOG_TAIL_BYTES = 4096;

export function initialState() {
  return { key: null, seq: null, appended: 0, missed: 0 };
}

/// One poll: reads the stats file and appends its object when it is new. Returns the next state.
export function recordOnce(state, statsPath, historyPath) {
  let text;
  try {
    text = fs.readFileSync(statsPath, 'utf8');
  } catch {
    return state;
  }
  const record = parseStatsText(text);
  if (!record) return state;
  const key = record.seq !== null ? `seq ${record.seq}` : `frame ${record.frame}`;
  if (key === state.key) return state;
  fs.appendFileSync(historyPath, `${JSON.stringify(record.raw)}\n`);
  const jump = record.seq !== null && state.seq !== null && record.seq > state.seq + 1 ? record.seq - state.seq - 1 : 0;
  return { key, seq: record.seq ?? state.seq, appended: state.appended + 1, missed: state.missed + jump };
}

/// Whether the log's tail carries the load-end line.
function gameEnded(logPath) {
  let fd;
  try {
    fd = fs.openSync(logPath, 'r');
    const size = fs.fstatSync(fd).size;
    const length = Math.min(size, LOG_TAIL_BYTES);
    const buffer = Buffer.alloc(length);
    fs.readSync(fd, buffer, 0, length, size - length);
    return LOAD_END_RE.test(buffer.toString('utf8'));
  } catch {
    return false;
  } finally {
    if (fd !== undefined) fs.closeSync(fd);
  }
}

/// Polls until the game has ended (or a signal), then reads once more. Resolves with the exit code.
export function watch(statsPath, historyPath, { log, pollMs = DEFAULT_POLL_MS } = {}) {
  return new Promise((resolve) => {
    let state = initialState();
    let timer = null;
    const finish = () => {
      if (timer === null) return;
      clearInterval(timer);
      timer = null;
      state = recordOnce(state, statsPath, historyPath);
      console.log(
        `stats recorder: ${state.appended} stats objects kept${state.missed ? `, ${state.missed} missed between polls` : ''}`,
      );
      process.removeListener('SIGINT', finish);
      process.removeListener('SIGTERM', finish);
      resolve(0);
    };
    const tick = () => {
      state = recordOnce(state, statsPath, historyPath);
      if (log && gameEnded(log)) finish();
    };
    process.once('SIGINT', finish);
    process.once('SIGTERM', finish);
    timer = setInterval(tick, pollMs);
    tick();
  });
}

function usage() {
  console.error('usage: node harness/stats-recorder.mjs <stats.json> <stats.jsonl> --log <run log> [--poll-ms 250]');
  return 2;
}

async function main(argv) {
  const positional = [];
  let log = null;
  let pollMs = DEFAULT_POLL_MS;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--log') log = argv[++i];
    else if (arg === '--poll-ms') pollMs = Number(argv[++i]);
    else if (arg.startsWith('--')) return usage();
    else positional.push(arg);
  }
  if (positional.length !== 2 || !log || !Number.isFinite(pollMs) || pollMs <= 0) return usage();
  return watch(positional[0], positional[1], { log, pollMs });
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  main(process.argv.slice(2))
    .then((code) => process.exit(code))
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}
