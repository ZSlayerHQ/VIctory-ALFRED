import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { LOCAL_ENV_NAMES, loadLocalEnv, parseLocalEnv } from './local-env.mjs';
import { captureFolderFromLog, capturesRoot } from './presentmon.mjs';
import { DEFAULT_SAVE_BACKUPS, saveBackupsRoot } from './save-snapshot.mjs';
import { NO_LOCAL_ENV } from './test-isolation.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const BS = String.fromCharCode(92);

function tempFile(t, name, text) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-env-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const file = path.join(dir, name);
  fs.writeFileSync(file, text);
  return { dir, file };
}

// An environment with none of the names local.env may set, so the machine's own settings cannot leak into a test.
function bareEnv(extra = {}) {
  const env = { ...process.env };
  for (const name of LOCAL_ENV_NAMES) delete env[name];
  delete env.VICTORY_LOCAL_ENV;
  return { ...env, ...extra };
}

test('parse: comments, blanks, export, quotes, spaces, = and backslashes in a value', () => {
  const text = [
    '# this machine',
    '',
    'GAME_EXE="D:/dump/Some Title/game.exe"',
    "export VICTORY_CAPTURES='D:/caps'",
    `VICTORY_BUILDS=D:${BS}builds${BS}`,
    'VICTORY_REPO=C:/x=y/victory   ',
  ].join('\r\n');
  const { values, warnings } = parseLocalEnv(text);
  assert.deepEqual(values, {
    GAME_EXE: 'D:/dump/Some Title/game.exe',
    VICTORY_CAPTURES: 'D:/caps',
    VICTORY_BUILDS: `D:${BS}builds${BS}`,
    VICTORY_REPO: 'C:/x=y/victory',
  });
  assert.deepEqual(warnings, []);
});

test('parse: a name outside the list is ignored, and the warning gives the line and name, never the value', () => {
  const { values, warnings } = parseLocalEnv('PATH=C:/secret-folder\nGAME_EXE=D:/g.exe\n');
  assert.deepEqual(values, { GAME_EXE: 'D:/g.exe' });
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /line 1/);
  assert.match(warnings[0], /PATH/);
  assert.doesNotMatch(warnings[0], /secret-folder/);
});

test('parse: VICTORY_ALLOW_OBS can never come from the file (D-025)', () => {
  const { values, warnings } = parseLocalEnv('VICTORY_ALLOW_OBS=1\n');
  assert.deepEqual(values, {});
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /VICTORY_ALLOW_OBS/);
  assert.equal(LOCAL_ENV_NAMES.includes('VICTORY_ALLOW_OBS'), false);
});

test('parse: the overlay path is a name the file may set', () => {
  const { values, warnings } = parseLocalEnv('VICTORY_PRECOMPILE_JSON=D:/stream/precompile.json\n');
  assert.deepEqual(values, { VICTORY_PRECOMPILE_JSON: 'D:/stream/precompile.json' });
  assert.equal(warnings.length, 0);
});

test('parse: a malformed line is skipped with its line number only', () => {
  const { values, warnings } = parseLocalEnv('just a path C:/private\n9BAD=x\n');
  assert.deepEqual(values, {});
  assert.equal(warnings.length, 2);
  assert.match(warnings[0], /line 1/);
  assert.match(warnings[1], /line 2/);
  for (const w of warnings) assert.doesNotMatch(w, /private/);
});

test('load: the environment wins, an empty variable counts as unset, and the applied names come back', (t) => {
  const { file } = tempFile(t, 'local.env', 'GAME_EXE=D:/file.exe\nVICTORY_CAPTURES=D:/caps\nVICTORY_BUILDS=D:/builds\n');
  const env = { GAME_EXE: 'C:/from-env.exe', VICTORY_CAPTURES: '' };
  const applied = loadLocalEnv({ file, env, warn: () => {} });
  assert.deepEqual(applied.sort(), ['VICTORY_BUILDS', 'VICTORY_CAPTURES']);
  assert.equal(env.GAME_EXE, 'C:/from-env.exe');
  assert.equal(env.VICTORY_CAPTURES, 'D:/caps');
  assert.equal(env.VICTORY_BUILDS, 'D:/builds');
});

test('load: no file is no change and no error', () => {
  const env = {};
  assert.deepEqual(loadLocalEnv({ file: path.join(os.tmpdir(), 'no-such-dir-xyz', 'local.env'), env }), []);
  assert.deepEqual(env, {});
});

test('load: VICTORY_LOCAL_ENV names the file', (t) => {
  const { file } = tempFile(t, 'other.env', 'PRESENTMON_EXE=D:/pm.exe\n');
  const env = { VICTORY_LOCAL_ENV: file };
  assert.deepEqual(loadLocalEnv({ env }), ['PRESENTMON_EXE']);
  assert.equal(env.PRESENTMON_EXE, 'D:/pm.exe');
});

test('shell loader: values arrive literally, the environment wins, and nothing is expanded', (t) => {
  const value = `D:/My Games/a $HOME b${BS}c/game.exe`;
  const { file } = tempFile(t, 'local.env', `GAME_EXE=${value}\nVICTORY_CAPTURES=D:/caps\nVICTORY_BUILDS=D:/b\n`);
  const script = [
    `. "${here.split(BS).join('/')}/local-env.sh"`,
    `load_local_env "${here.split(BS).join('/')}"`,
    'printf "%s\\n" "$GAME_EXE" "$VICTORY_CAPTURES" "$VICTORY_BUILDS"',
  ].join('\n');
  const result = spawnSync('bash', ['-c', script], {
    encoding: 'utf8',
    env: bareEnv({ VICTORY_LOCAL_ENV: file, VICTORY_BUILDS: 'C:/env-wins' }),
  });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(result.stdout.split(/\r?\n/).slice(0, 3), [value, 'D:/caps', 'C:/env-wins']);
});

test('run-world.sh builds the capture folder under VICTORY_CAPTURES', (t) => {
  const { dir } = tempFile(t, 'unused', '');
  const caps = path.join(dir, 'caps').split(BS).join('/');
  const result = spawnSync('bash', [path.join(here, 'run-world.sh'), '--tag', 'localenvtest', '--no-record'], {
    encoding: 'utf8',
    env: bareEnv({
      VICTORY_LOCAL_ENV: path.join(dir, 'none.env'),
      VICTORY_CAPTURES: caps,
      VICTORY_BIN: path.join(dir, 'no-such-victory.exe'),
      // run-world.sh refuses a direct run that would skip the lock; this test is about the capture folder.
      WORLD_RUN_HOLDS_LOCK: '1',
    }),
  });
  assert.equal(result.status, 2, 'stops at the missing binary');
  const made = fs.readdirSync(caps);
  assert.equal(made.length, 1);
  assert.match(made[0], /^\d{8}-\d{6}-localenvtest$/);
});

test('capture root: VICTORY_CAPTURES, else the generic relative default, used for the log-name fallback', () => {
  assert.equal(capturesRoot({}), 'victory-captures');
  assert.equal(capturesRoot({ VICTORY_CAPTURES: `D:${BS}caps${BS}` }), `D:${'/'}caps`);
  assert.equal(
    captureFolderFromLog(['harness build: abc'], 'C:/r/harness/logs/20261124-120000-beta.log', 'D:/caps/'),
    'D:/caps/20261124-120000-beta',
  );
  // A capture: line still wins over any root.
  assert.equal(
    captureFolderFromLog(['capture: D:/caps/20261124-120000-alpha'], 'x/20261124-120000-beta.log', 'D:/caps'),
    'D:/caps/20261124-120000-alpha',
  );
});

test('save backups root: VICTORY_SAVE_BACKUPS, else the default beside the user home', () => {
  assert.equal(saveBackupsRoot({}), DEFAULT_SAVE_BACKUPS);
  assert.equal(saveBackupsRoot({ VICTORY_SAVE_BACKUPS: `D:${BS}saves${BS}` }), `D:${'/'}saves`);
});

test('a hostile local.env cannot reach a test that points the loader away from it', (t) => {
  // A file that does what this machine's does: a captures root holding a matching PresentMon CSV. If the loader
  // read it, the experiment table would gain the PresentMon columns. The suite's spawns use isolatedEnv(), which
  // points the loader at NO_LOCAL_ENV; naming the hostile file explicitly must still prove the file is effective.
  const { dir } = tempFile(t, 'unused', '');
  const caps = path.join(dir, 'hostile-captures');
  const log = path.join(dir, '20261124-120000-alpha.log');
  const csv = path.join(caps, '20261124-120000-alpha', 'presentmon.csv');
  fs.mkdirSync(path.dirname(csv), { recursive: true });
  fs.writeFileSync(csv, 'Application,ProcessID,FrameTime,GPUBusy\r\nvictory.exe,100,10,2\r\nvictory.exe,100,20,4\r\n');
  fs.writeFileSync(log, '2026-11-24T12:00:00Z  INFO victory_gpu::frame: frame interval interval=33.333ms\n');
  const hostile = path.join(dir, 'local.env');
  fs.writeFileSync(hostile, `VICTORY_CAPTURES=${caps.split(BS).join('/')}\n`);

  const table = (env) =>
    spawnSync(process.execPath, [path.join(here, 'experiment-table.mjs'), log], { encoding: 'utf8', env });

  const isolated = table(bareEnv({ VICTORY_LOCAL_ENV: NO_LOCAL_ENV }));
  assert.equal(isolated.status, 0, isolated.stderr);
  assert.equal(isolated.stdout.includes('median frame ms'), false, 'the isolated run must not see the hostile root');

  const named = table(bareEnv({ VICTORY_LOCAL_ENV: hostile }));
  assert.equal(named.status, 0, named.stderr);
  assert.match(named.stdout, /median frame ms/, 'the file itself still works when a test names it');
});
