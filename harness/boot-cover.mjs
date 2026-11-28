// The loading cover: a video on top of the game's window from the start of a session until the world is on screen.
//
// The harness starts the player (ffplay: borderless, always on top, silent, looping) before the game, so the game's
// window opens under it and keeps the keyboard. This watcher reads the run's log as it grows and ends the player
// when a frame line shows the world drawn at a playable rate, when the ceiling passes, or when the player is
// already gone. It never touches the game or its sound.
//
//   node boot-cover.mjs watch <log> <player's Windows process id> [--ceiling-seconds N]
//
// Prints one `harness: loading cover ended ...` line and exits 0.
//
// Why a window and not a pass in the runtime: while the world loads the title presents about three frames a second
// (2026-11-28: 3.0 to 3.6 fps for 50 s), and a video drawn in its frames would stutter for that whole stretch.
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { stripAnsi } from './parse-run.mjs';

// The runtime's frame line, as the base writes it: "<n> fps, <n> ms/frame (...), <n> draws/frame, dropped ...".
const FRAME_RE = /victory_gpu::host: ([0-9.]+) fps, [0-9.]+ ms\/frame.*?\b([0-9]+) draws\/frame/;

// The world is on screen: thousands of draws at a rate a person can play. The loading screens draw about a dozen
// things at hundreds of frames a second; the world while it loads draws thousands at three.
export const WORLD = { minFps: 8, minDraws: 1000 };

export function frameOf(line) {
  const found = stripAnsi(line).match(FRAME_RE);
  return found ? { fps: Number(found[1]), draws: Number(found[2]) } : null;
}

export function worldArrived(line, rule = WORLD) {
  const frame = frameOf(line);
  return frame !== null && frame.fps >= rule.minFps && frame.draws >= rule.minDraws;
}

function alive(pid) {
  const out = spawnSync('tasklist', ['/FI', `PID eq ${pid}`, '/NH'], { encoding: 'utf8' }).stdout || '';
  return new RegExp(`\\b${pid}\\b`).test(out);
}

function end(pid) {
  spawnSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore' });
}

/// Follows `log` until the world arrives, the ceiling passes or the player is gone; ends the player; returns why.
export async function watch({ log, pid, ceilingSeconds = 600, pollMs = 500, isAlive = alive, endPlayer = end }) {
  const started = Date.now();
  let offset = 0;
  let rest = '';
  for (;;) {
    const waited = (Date.now() - started) / 1000;
    if (!isAlive(pid)) return { why: 'the player was already closed', waited };
    if (fs.existsSync(log)) {
      const size = fs.statSync(log).size;
      if (size > offset) {
        const fd = fs.openSync(log, 'r');
        const buffer = Buffer.alloc(size - offset);
        fs.readSync(fd, buffer, 0, buffer.length, offset);
        fs.closeSync(fd);
        offset = size;
        const lines = (rest + buffer.toString('utf8')).split(/\r?\n/);
        rest = lines.pop();
        for (const line of lines) {
          if (worldArrived(line)) {
            const { fps, draws } = frameOf(line);
            endPlayer(pid);
            return { why: `the world is on screen (${fps} fps, ${draws} draws a frame)`, waited };
          }
        }
      }
    }
    if (waited >= ceilingSeconds) {
      endPlayer(pid);
      return { why: `the world had not come after ${ceilingSeconds} s`, waited };
    }
    await new Promise((resolve) => setTimeout(resolve, pollMs));
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === fs.realpathSync(process.argv[1])) {
  const [command, log, pid, ...rest] = process.argv.slice(2);
  const at = rest.indexOf('--ceiling-seconds');
  const ceilingSeconds = at >= 0 ? Number(rest[at + 1]) : 600;
  if (command !== 'watch' || !log || !/^\d+$/.test(pid || '') || !(ceilingSeconds > 0)) {
    console.error("usage: node boot-cover.mjs watch <log> <player's process id> [--ceiling-seconds N]");
    process.exit(2);
  }
  const { why, waited } = await watch({ log, pid: Number(pid), ceilingSeconds });
  console.log(`harness: loading cover ended after ${Math.round(waited)} s: ${why}`);
}
