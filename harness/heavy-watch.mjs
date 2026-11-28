#!/usr/bin/env node
// Ends a VIctory world run N seconds after its heavy 3D scene begins.
//
//   node harness/heavy-watch.mjs <log> <seconds> [--poll-ms 250]
//
// The heavy scene starts at the first VIctory frame line whose draws plus
// dropped draws per frame reaches 1000 (findTrigger in presentmon.mjs, the rule
// PresentMon starts on too). N seconds after that line's timestamp the watcher
// ends this run's victory.exe, appends
//   harness: ended by --heavy-seconds after N s of heavy scene
// to the log, and exits. If the game exits by itself first (a device removal,
// a crash) it does nothing and exits quietly.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { endSelected } from './end-run.mjs';

import { findTrigger } from './presentmon.mjs';
import { stripAnsi } from './parse-run.mjs';

export const DEFAULT_POLL_MS = 250;
const HERE = path.dirname(fileURLToPath(import.meta.url));

export const endedMarker = (seconds) =>
  `harness: ended by --heavy-seconds after ${seconds} s of heavy scene`;

// The watcher's decision. `lines` are the lines read since the last poll;
// `triggerAt` is kept from the poll that first saw the trigger, so a later poll
// without that line still ends at the right time. `gameAlive` is checked first:
// a game that is gone is nothing to end, so the watcher says nothing.
export function decide(lines, { seconds, now = Date.now(), gameAlive = true, triggerAt = null } = {}) {
  const hit = findTrigger(lines);
  let at = triggerAt;
  if (hit && at === null) {
    const m = stripAnsi(lines[hit.index]).match(/^(\S+Z)\s/);
    at = m ? Date.parse(m[1]) : null;
    if (Number.isNaN(at)) at = null;
  }
  if (!hit && at === null)
    return { action: gameAlive ? 'wait' : 'gone', trigger: null, elapsedSeconds: null };
  const elapsedSeconds = at === null ? 0 : Math.max(0, (now - at) / 1000);
  if (!gameAlive) return { action: 'gone', trigger: hit, elapsedSeconds };
  if (elapsedSeconds >= seconds) return { action: 'end', trigger: hit, elapsedSeconds };
  return { action: 'wait', trigger: hit, elapsedSeconds };
}

// run-process.ps1's strict selection, one pid per line. null means PowerShell
// failed, which is not evidence that the game is gone.
function selectionPids(psFile) {
  const result = spawnSync(
    'powershell.exe',
    ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', psFile, '-Pids', '-Strict'],
    { encoding: 'utf8', timeout: 15_000, windowsHide: true },
  );
  if (result.error || result.status !== 0) return null;
  return (result.stdout || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => /^\d+$/.test(line))
    .map(Number);
}

function isAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err.code === 'EPERM'; // It exists, but is not ours to signal.
  }
}

// taskkill needs /F: a plain taskkill asks the window to close, and this title ignores that.
function endPid(pid) {
  const result = spawnSync('taskkill', ['/F', '/PID', String(pid)], {
    encoding: 'utf8',
    windowsHide: true,
  });
  return result.status === 0;
}

export function watch(logPath, seconds, options = {}) {
  const pollMs = options.pollMs ?? DEFAULT_POLL_MS;
  const psFile = options.psFile ?? path.join(HERE, 'run-process.ps1');
  return new Promise((resolve) => {
    let pos = 0;
    let pending = '';
    let trigger = null;
    let triggerAt = null;
    let pids = null; // This run's pids; null until the selection (or PowerShell) says.
    let seenGame = false; // A pid has been seen at least once.
    let lastSelection = 0;
    let stopping = false;
    let timer = null;

    const stop = (code) => {
      if (stopping) return;
      stopping = true;
      if (timer) clearInterval(timer);
      resolve(code);
    };

    const refreshPids = () => {
      lastSelection = Date.now();
      const found = selectionPids(psFile);
      if (found === null) return;
      pids = found;
      if (found.length) seenGame = true;
    };

    const gameAlive = () => {
      if (!seenGame) return true; // Still starting: not yet evidence of a game, let alone a gone one.
      if (!pids) return true;
      if (pids.some(isAlive)) return true;
      // The cached pid died. Re-select: "gone" must mean no process of this run is left.
      refreshPids();
      return pids ? pids.some(isAlive) : true;
    };

    const readNew = () => {
      let size;
      try {
        size = fs.statSync(logPath).size;
      } catch {
        return; // The harness creates the log before it starts us; give it a moment.
      }
      if (size < pos) {
        pos = 0;
        pending = '';
      }
      if (size === pos) return;
      let data;
      try {
        const fd = fs.openSync(logPath, 'r');
        try {
          const buffer = Buffer.alloc(size - pos);
          fs.readSync(fd, buffer, 0, buffer.length, pos);
          data = buffer.toString('utf8');
        } finally {
          fs.closeSync(fd);
        }
      } catch {
        return;
      }
      pos = size;
      pending += data;
      const lines = pending.split('\n');
      pending = lines.pop();
      if (trigger) return;
      const hit = findTrigger(lines);
      if (!hit) return;
      trigger = { draws: hit.draws, dropped: hit.dropped };
      const m = stripAnsi(lines[hit.index]).match(/^(\S+Z)\s/);
      const at = m ? Date.parse(m[1]) : NaN;
      triggerAt = Number.isNaN(at) ? Date.now() : at;
    };

    const tick = () => {
      if (stopping) return;
      readNew();
      if (!seenGame && Date.now() - lastSelection >= 1000) refreshPids();

      if (!trigger) {
        if (seenGame && !gameAlive()) stop(0);
        return;
      }

      const decision = decide([], {
        seconds,
        now: Date.now(),
        gameAlive: gameAlive(),
        triggerAt,
      });
      if (decision.action === 'gone') {
        stop(0);
        return;
      }
      if (decision.action !== 'end') return;

      // A fresh selection at the moment of the kill, never the cached ids (end-run.mjs says why).
      const result = endSelected(() => selectionPids(psFile), endPid);
      if (result.fresh) pids = result.fresh;
      if (result.outcome === 'gone') {
        // Nothing of this run is left: it ended by itself at the same moment.
        stop(0);
        return;
      }
      if (result.outcome !== 'ended') return; // PowerShell failed, or taskkill did: retry next poll.
      try {
        fs.appendFileSync(logPath, endedMarker(seconds) + '\n');
      } catch {
        // The console line below still says what happened; never take the watcher down over the log.
      }
      console.log(endedMarker(seconds));
      stop(0);
    };

    process.once('SIGINT', () => stop(0));
    process.once('SIGTERM', () => stop(0));
    refreshPids();
    timer = setInterval(tick, pollMs);
    tick();
  });
}

function usage() {
  console.error('usage: node heavy-watch.mjs <log> <seconds> [--poll-ms 250]');
}

async function main(argv) {
  const positional = [];
  let pollMs = DEFAULT_POLL_MS;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--poll-ms') pollMs = Number(argv[++i]);
    else if (arg.startsWith('--')) {
      console.error(`heavy-watch: unknown argument ${arg}`);
      usage();
      return 2;
    } else {
      positional.push(arg);
    }
  }
  const seconds = Number(positional[1]);
  if (
    positional.length !== 2 ||
    !Number.isFinite(seconds) ||
    seconds < 0 ||
    !Number.isFinite(pollMs) ||
    pollMs <= 0
  ) {
    usage();
    return 2;
  }
  return watch(positional[0], seconds, { pollMs });
}

const isMain =
  process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  main(process.argv.slice(2))
    .then((code) => process.exit(code))
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}
