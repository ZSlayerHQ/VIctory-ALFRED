import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { isolatedEnv } from './test-isolation.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const wrapper = path.join(here, 'world-run.sh');
const RUN_SCRIPTS = ['run-world.sh', 'local-env.sh', 'public-lines.sh', 'pipeline-library.sh', 'obs.sh'];

function tempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'world-run-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

test('world-run copies the scripts it runs and the files it sources into the capture folder, and runs the copy', (t) => {
  const dir = tempDir(t);
  const caps = path.join(dir, 'captures');
  const lock = path.join(dir, 'run.lock');
  const tag = `copytest-${process.pid}-${Date.now()}`;
  fs.writeFileSync(path.join(dir, 'victory.toml'), '');
  const bash = spawnSync('bash', ['-c', 'command -v bash'], { encoding: 'utf8' }).stdout.trim();
  const result = spawnSync('bash', [wrapper, '--tag', tag, '--print-env'], {
    encoding: 'utf8',
    env: isolatedEnv({
      VICTORY_BIN: bash,
      VICTORY_DIR: dir,
      VICTORY_CONFIG: path.join(dir, 'victory.toml'),
      GAME_EXE: path.join(dir, 'game.exe'),
      VICTORY_CAPTURES: caps,
      VICTORY_RUN_LOCK: lock,
      VICTORY_D3D12_WARP: '1',
    }),
  });
  assert.equal(result.status, 0, result.stderr);
  const made = fs.readdirSync(caps).filter((name) => name.endsWith(`-${tag}`));
  assert.equal(made.length, 1, result.stdout);
  const capture = path.join(caps, made[0]);
  for (const file of RUN_SCRIPTS) {
    const copy = path.join(capture, file);
    assert.equal(fs.existsSync(copy), true, `${file} was copied`);
    assert.equal(fs.readFileSync(copy, 'utf8'), fs.readFileSync(path.join(here, file), 'utf8'), `${file} is a copy`);
  }
  // The copy itself ran: only run-world.sh writes these two.
  assert.equal(fs.existsSync(path.join(capture, 'game-args.txt')), true);
  assert.equal(fs.existsSync(path.join(capture, 'env.txt')), true);
  assert.equal(fs.existsSync(lock), false, 'the run lock is gone after the run');
});

test('--require-quiet passes through the wrapper, and --print-env still starts nothing', (t) => {
  const dir = tempDir(t);
  const lock = path.join(dir, 'run.lock');
  fs.writeFileSync(path.join(dir, 'victory.toml'), '');
  const bash = spawnSync('bash', ['-c', 'command -v bash'], { encoding: 'utf8' }).stdout.trim();
  const result = spawnSync('bash', [wrapper, '--require-quiet', '--print-env'], {
    encoding: 'utf8',
    env: isolatedEnv({
      VICTORY_BIN: bash,
      VICTORY_DIR: dir,
      VICTORY_CONFIG: path.join(dir, 'victory.toml'),
      GAME_EXE: path.join(dir, 'game.exe'),
      VICTORY_CAPTURES: path.join(dir, 'captures'),
      VICTORY_RUN_LOCK: lock,
      VICTORY_D3D12_WARP: '1',
    }),
  });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(fs.existsSync(lock), false);
});

test('a run started through world-run holds the lock, so the copy is allowed to start', (t) => {
  const dir = tempDir(t);
  const lock = path.join(dir, 'run.lock');
  fs.writeFileSync(path.join(dir, 'victory.toml'), '');
  const bash = spawnSync('bash', ['-c', 'command -v bash'], { encoding: 'utf8' }).stdout.trim();
  const result = spawnSync('bash', [wrapper, '--print-env'], {
    encoding: 'utf8',
    env: isolatedEnv({
      VICTORY_BIN: bash,
      VICTORY_DIR: dir,
      VICTORY_CONFIG: path.join(dir, 'victory.toml'),
      GAME_EXE: path.join(dir, 'game.exe'),
      VICTORY_CAPTURES: path.join(dir, 'captures'),
      VICTORY_RUN_LOCK: lock,
      VICTORY_D3D12_WARP: '1',
    }),
  });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(fs.existsSync(lock), false, 'the lock is gone after the run');
});

test("the lock's owner file names the run's numeric pid and a token that starts with it, which the run is given", (t) => {
  // A copy of the wrapper beside a stand-in run-world.sh that prints the owner file and the token it was given.
  const dir = tempDir(t);
  for (const file of ['world-run.sh', 'local-env.sh', 'local-env.mjs']) fs.copyFileSync(path.join(here, file), path.join(dir, file));
  for (const file of RUN_SCRIPTS.filter((f) => f !== 'run-world.sh' && f !== 'local-env.sh')) fs.writeFileSync(path.join(dir, file), '');
  fs.writeFileSync(
    path.join(dir, 'run-world.sh'),
    'cat "$VICTORY_RUN_LOCK/owner"\nprintf "given %s\n" "$WORLD_RUN_LOCK_TOKEN"\n',
  );
  const lock = path.join(dir, 'run.lock');
  const result = spawnSync('sh', [path.join(dir, 'world-run.sh')], {
    encoding: 'utf8',
    env: isolatedEnv({ VICTORY_RUN_LOCK: lock, VICTORY_CAPTURES: path.join(dir, 'captures'), VICTORY_D3D12_WARP: '1' }),
  });
  assert.equal(result.status, 0, result.stderr);
  const pid = result.stdout.match(/^pid (\S+), started /m);
  assert.ok(pid, result.stdout);
  assert.match(pid[1], /^[0-9]+$/, `the owner line names a numeric pid, not ${pid[1]}`);
  const token = result.stdout.match(/^token (\S+)$/m);
  assert.ok(token, result.stdout);
  assert.ok(token[1].startsWith(`${pid[1]}-`), `the token ${token[1]} starts with the pid ${pid[1]}`);
  assert.match(result.stdout, new RegExp(`^given ${token[1]}$`, 'm'), 'the run is given the same token');
  assert.equal(fs.existsSync(lock), false, 'the lock is gone after the run');
});
