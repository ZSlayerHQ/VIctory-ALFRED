import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { frameOf, watch, worldArrived } from './boot-cover.mjs';

// Frame lines in the base's own shape, at the three stages of a start (2026-11-28, run public-noone).
const line = (fps, ms, draws) =>
  `2026-11-28T02:38:19.000000Z  INFO victory_gpu::host: ${fps} fps, ${ms} ms/frame (45.3 preparing: copies 23.0 ms 57.4 MB, textures 3.2 ms 74.8 MB; 0.0 on the GPU; 2.1 host waits), ${draws} draws/frame, dropped 353 without a host pipeline (0 being made) and 1084 indirect; heap buffers 2 passes 0.1 ms`;
const LOADING_SCREEN = line('469.2', '2.1', 12);
const WORLD_LOADING = line('3.6', '275.4', 9672);
const WORLD = line('18.0', '55.6', 6933);

test('a frame line gives its rate and its draws, with or without colour codes', () => {
  assert.deepEqual(frameOf(WORLD), { fps: 18, draws: 6933 });
  const coloured = WORLD.replace('INFO', '\x1b[32m INFO\x1b[0m').replace('victory_gpu::host', '\x1b[2mvictory_gpu::host\x1b[0m\x1b[2m');
  assert.deepEqual(frameOf(coloured.replace('::host\x1b[0m\x1b[2m:', '::host\x1b[0m\x1b[2m:\x1b[0m')), { fps: 18, draws: 6933 });
  assert.equal(frameOf('2026-11-28T02:38:19Z  INFO victory_gpu::pipeline: first bind pipeline=0x1'), null);
});

test('the world has arrived only when it is drawn at a playable rate', () => {
  assert.equal(worldArrived(LOADING_SCREEN), false, 'a dozen draws at hundreds of frames a second is a loading screen');
  assert.equal(worldArrived(WORLD_LOADING), false, 'thousands of draws at three frames a second is still loading');
  assert.equal(worldArrived(WORLD), true);
});

test('the watcher ends the player when the world arrives, and not before', async (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'boot-cover-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const log = path.join(dir, 'run.log');
  fs.writeFileSync(log, `${LOADING_SCREEN}\n${WORLD_LOADING}\n`);
  let ended = 0;
  const watching = watch({ log, pid: 1, pollMs: 20, isAlive: () => true, endPlayer: () => (ended += 1) });
  await new Promise((resolve) => setTimeout(resolve, 120));
  assert.equal(ended, 0, 'still loading: the cover stays');
  fs.appendFileSync(log, `${WORLD.slice(0, 60)}`); // a line cut mid-write is not read until it is whole
  await new Promise((resolve) => setTimeout(resolve, 80));
  assert.equal(ended, 0);
  fs.appendFileSync(log, `${WORLD.slice(60)}\n`);
  const result = await watching;
  assert.equal(ended, 1);
  assert.match(result.why, /the world is on screen \(18 fps, 6933 draws a frame\)/);
});

test('the watcher gives up at its ceiling, and stops at once when the player is already closed', async (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'boot-cover-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const log = path.join(dir, 'run.log');
  fs.writeFileSync(log, `${WORLD_LOADING}\n`);
  let ended = 0;
  const late = await watch({ log, pid: 1, ceilingSeconds: 0.1, pollMs: 20, isAlive: () => true, endPlayer: () => (ended += 1) });
  assert.equal(ended, 1);
  assert.match(late.why, /had not come/);
  const gone = await watch({ log, pid: 1, pollMs: 20, isAlive: () => false, endPlayer: () => (ended += 1) });
  assert.equal(ended, 1, 'nothing to end');
  assert.match(gone.why, /already closed/);
});
