import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { isolatedEnv } from './test-isolation.mjs';
import { copyVerified, dataDir, manifest, runtimeDataDir, measure, oneFolder, plainName, processRunning, restore, saveAs, saveFolder } from './save-snapshot.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'save-snapshot.mjs');
const GUID = '0f0f0f0f-0000-4000-8000-000000000000';
// No game is ever running for these tests: the check of a real one is its own test.
const noGame = () => false;

function runSnapshot(args) {
  // A process name no machine runs, so the command line never refuses for a game that happens to be up.
  return spawnSync(process.execPath, [script, ...args, '--game-process', 'no-game-for-save-tests.exe'], { encoding: 'utf8', env: isolatedEnv({ VICTORY_DATA_DIR: '' }) });
}

function writeFile(file, text) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text);
}

// A data folder laid out as the runtime makes it, with one container in the config folder.
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'victory-saves-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const data = path.join(root, 'data');
  const folder = path.join(data, '0000ABCD', 'saves', 'feedfacefeedface', GUID);
  writeFile(path.join(folder, 'box-one~0123456789abcdef', 'v1', 'manifest'), 'a-manifest');
  writeFile(path.join(folder, 'box-one~0123456789abcdef', 'v1', '0'), 'save-data');
  writeFile(path.join(folder, 'box-one~0123456789abcdef', 'v1', '1'), 'save-header');
  const toml = path.join(root, 'victory.toml');
  writeFile(toml, `[paths]\ndata_dir = '${data}'\n`);
  const snapshot = path.join(root, 'snapshots', 'example-start');
  writeFile(path.join(snapshot, 'Player~fedcba9876543210', 'v3', 'manifest'), 'other-manifest');
  writeFile(path.join(snapshot, 'Player~fedcba9876543210', 'v3', '0'), 'other-data');
  return { root, data, folder, toml, snapshot, snapshots: path.join(root, 'snapshots'), before: path.join(root, 'before') };
}

test('the config gives the save folder: one title, one user, one config, or a refusal naming what is there', (t) => {
  const f = fixture(t);
  assert.equal(dataDir(f.toml), f.data);
  assert.equal(saveFolder(f.toml), f.folder);
  fs.mkdirSync(path.join(f.data, '0000ABCD', 'saves', '0009000000000002'));
  assert.throws(() => saveFolder(f.toml), /expected exactly one user folder/);
  const second = fixture(t);
  fs.mkdirSync(path.join(second.folder, '..', '11111111-1111-1111-1111-111111111111'));
  assert.throws(() => saveFolder(second.toml), /expected exactly one config folder/);
  assert.throws(() => oneFolder(path.join(f.root, 'nope'), 'title folder'), /cannot list/);
});

test('a restore backs up the old save and replaces it with the snapshot, both verified', (t) => {
  const f = fixture(t);
  const old = measure(f.folder);
  const done = restore('example-start', {
    config: f.toml,
    env: {},
    isRunning: noGame,
    snapshots: f.snapshots,
    before: f.before,
    lock: path.join(f.root, 'absent.lock'),
    stamp: 'stamp',
  });
  assert.equal(done.folder, f.folder);
  assert.equal(done.backup, path.join(f.before, 'stamp-example-start'));
  const now = measure(f.folder);
  const want = measure(f.snapshot);
  assert.deepEqual(now, want);
  assert.deepEqual(measure(done.backup), old, 'the backup holds what was there before');
  assert.deepEqual(fs.readdirSync(f.folder), ['Player~fedcba9876543210']);
  assert.equal(
    fs.readFileSync(path.join(f.folder, 'Player~fedcba9876543210', 'v3', '0'), 'utf8'),
    'other-data',
  );
});

test('a restore refuses before touching the saves: a held lock, an unknown name, an existing backup', (t) => {
  const f = fixture(t);
  const before = measure(f.folder);
  const lock = path.join(f.root, 'run.lock');
  writeFile(lock, '');
  assert.throws(
    () => restore('example-start', { config: f.toml, env: {}, isRunning: noGame, snapshots: f.snapshots, before: f.before, lock }),
    /a run holds the harness lock/,
  );
  assert.throws(
    () => restore('nowhere', { config: f.toml, env: {}, isRunning: noGame, snapshots: f.snapshots, before: f.before, lock: path.join(f.root, 'a.lock') }),
    /no snapshot 'nowhere'; snapshots: example-start/,
  );
  writeFile(path.join(f.before, 'taken-example-start', 'keep'), 'x');
  assert.throws(
    () => restore('example-start', { config: f.toml, env: {}, isRunning: noGame, snapshots: f.snapshots, before: f.before, lock: path.join(f.root, 'a.lock'), stamp: 'taken' }),
    /already exists: refusing to overwrite/,
  );
  assert.deepEqual(measure(f.folder), before, 'a refusal leaves the save folder as it was');
});

test('a backup that cannot be made stops the restore with the saves untouched', (t) => {
  const f = fixture(t);
  const before = measure(f.folder);
  const notAFolder = path.join(f.root, 'before-is-a-file');
  writeFile(notAFolder, 'x');
  assert.throws(
    () => restore('example-start', { config: f.toml, env: {}, isRunning: noGame, snapshots: f.snapshots, before: notAFolder, lock: path.join(f.root, 'a.lock') }),
    /does not match|cannot list|ENOTDIR|EEXIST|not a directory|exist/i,
  );
  assert.deepEqual(measure(f.folder), before);
});

test('save-as makes a new snapshot of the folder and refuses an existing name', (t) => {
  const f = fixture(t);
  const done = saveAs('captured', { config: f.toml, env: {}, isRunning: noGame, snapshots: f.snapshots });
  assert.equal(done.snapshot, path.join(f.snapshots, 'captured'));
  assert.deepEqual(measure(done.snapshot), measure(f.folder));
  assert.throws(() => saveAs('captured', { config: f.toml, env: {}, isRunning: noGame, snapshots: f.snapshots }), /already exists/);
  assert.throws(() => saveAs('example-start', { config: f.toml, env: {}, isRunning: noGame, snapshots: f.snapshots }), /already exists/);
  assert.throws(() => saveAs('../escape', { config: f.toml, env: {}, isRunning: noGame, snapshots: f.snapshots }), /not a plain snapshot name/);
});

test('the command line restores and saves with the paths it is given', (t) => {
  const f = fixture(t);
  const lock = path.join(f.root, 'absent.lock');
  let result = runSnapshot([
    'restore',
    'example-start',
    '--config',
    f.toml,
    '--snapshots',
    f.snapshots,
    '--before',
    f.before,
    '--lock',
    lock,
    '--stamp',
    'cli',
  ]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /restored 'example-start' into /);
  assert.ok(fs.existsSync(path.join(f.before, 'cli-example-start')));
  result = runSnapshot(['save-as', 'other', '--config', f.toml, '--snapshots', f.snapshots]);
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(measure(path.join(f.snapshots, 'other')), measure(f.folder));
  result = runSnapshot(['restore', 'missing', '--config', f.toml, '--snapshots', f.snapshots, '--before', f.before, '--lock', lock]);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /no snapshot 'missing'/);
});

test('a run that holds the lock itself restores under it', (t) => {
  const f = fixture(t);
  const lock = path.join(f.root, 'run.lock');
  writeFile(path.join(lock, 'owner'), 'pid 1, started now\ntoken own-run\n');
  const done = restore('example-start', {
    config: f.toml,
    env: {},
    isRunning: noGame,
    snapshots: f.snapshots,
    before: f.before,
    lock,
    callerHoldsLock: true,
    lockToken: 'own-run',
    stamp: 'own',
  });
  assert.deepEqual(measure(f.folder), measure(f.snapshot));
  assert.equal(done.backup, path.join(f.before, 'own-example-start'));
});

// The restore's guarantees: the live save is never left empty or half-written, a snapshot name is never a path, the
// folders never overlap, and the lock must be this run's.
const restoreIn = (f, extra = {}) => ({ config: f.toml, env: {}, isRunning: noGame, snapshots: f.snapshots, before: f.before, lock: path.join(f.root, 'a.lock'), ...extra });

test('an empty snapshot is refused and the save kept, unless --allow-empty asks for it', (t) => {
  const f = fixture(t);
  const before = measure(f.folder);
  fs.mkdirSync(path.join(f.snapshots, 'nothing'));
  assert.throws(() => restore('nothing', restoreIn(f)), /is empty: refusing to replace the save with nothing/);
  assert.deepEqual(measure(f.folder), before, 'the save is as it was');
  restore('nothing', restoreIn(f, { allowEmpty: true, stamp: 'empty' }));
  assert.deepEqual(measure(f.folder), { files: 0, bytes: 0 }, 'asked for, the empty snapshot is restored');
  assert.deepEqual(measure(path.join(f.before, 'empty-nothing')), before, 'with the old save in its backup');
});

test('a snapshot name is a plain name, never a path', (t) => {
  const f = fixture(t);
  const before = measure(f.folder);
  for (const name of ['../snapshots/example-start', 'a/b', '..', '.', 'x\\y']) {
    assert.throws(() => restore(name, restoreIn(f)), /is not a plain snapshot name/, name);
    assert.throws(() => saveAs(name, { config: f.toml, env: {}, isRunning: noGame, snapshots: f.snapshots }), /is not a plain snapshot name/, name);
  }
  assert.deepEqual(measure(f.folder), before);
});

test('overlapping save, snapshot and backup folders are refused before anything is copied', (t) => {
  const f = fixture(t);
  const before = measure(f.folder);
  assert.throws(() => restore('example-start', restoreIn(f, { before: path.join(f.folder, 'backups') })), /overlap: refusing/);
  const inside = path.join(f.folder, 'snaps');
  fs.cpSync(f.snapshot, path.join(inside, 'example-start'), { recursive: true });
  const withSnaps = measure(f.folder);
  assert.throws(() => restore('example-start', restoreIn(f, { snapshots: inside })), /overlap: refusing/);
  assert.throws(() => saveAs('again', { config: f.toml, env: {}, isRunning: noGame, snapshots: inside }), /overlap: refusing/);
  assert.deepEqual(measure(f.folder), withSnaps, 'nothing was copied into or out of the save');
  assert.ok(withSnaps.files > before.files);
});

test('a snapshot that fails to copy leaves the save as it was and no staging folder', (t) => {
  const f = fixture(t);
  const before = measure(f.folder);
  let calls = 0;
  const copy = (from, to) => {
    calls += 1;
    if (calls === 2) {
      fs.mkdirSync(to, { recursive: true });
      fs.writeFileSync(path.join(to, 'half'), 'x');
      throw new Error('the disk is full');
    }
    return copyVerified(from, to);
  };
  assert.throws(() => restore('example-start', restoreIn(f, { copy, stamp: 'fail' })), /stopped before the save was touched: the disk is full/);
  assert.deepEqual(measure(f.folder), before, 'the save is as it was');
  const parent = path.dirname(f.folder);
  assert.deepEqual(fs.readdirSync(parent), [path.basename(f.folder)], 'no staging or old folder is left beside it');
});

test('a swap that does not verify is rolled back to the old save', (t) => {
  const f = fixture(t);
  const before = measure(f.folder);
  let calls = 0;
  // The staging copy "succeeds" but holds an extra file, so the swapped-in folder does not match the snapshot.
  const copy = (from, to) => {
    calls += 1;
    const done = copyVerified(from, to);
    if (calls === 2) fs.writeFileSync(path.join(to, 'extra'), 'x');
    return done;
  };
  assert.throws(() => restore('example-start', restoreIn(f, { copy, stamp: 'swap' })), /was rolled back, the save is as it was/);
  assert.deepEqual(measure(f.folder), before, 'the old save is back');
  assert.deepEqual(fs.readdirSync(path.dirname(f.folder)), [path.basename(f.folder)], 'and nothing else is beside it');
});

test("a restore under another run's lock, or under no lock, is refused", (t) => {
  const f = fixture(t);
  const before = measure(f.folder);
  const lock = path.join(f.root, 'run.lock');
  writeFile(path.join(lock, 'owner'), 'pid 2, started now\ntoken other-run\n');
  assert.throws(() => restore('example-start', restoreIn(f, { lock, callerHoldsLock: true, lockToken: 'this-run' })), /held by another run/);
  assert.throws(() => restore('example-start', restoreIn(f, { lock: path.join(f.root, 'none.lock'), callerHoldsLock: true, lockToken: 'this-run' })), /is not held/);
  assert.throws(() => restore('example-start', restoreIn(f, { lock })), /a run holds the harness lock/, 'a held lock refuses a restore that would take it');
  assert.deepEqual(measure(f.folder), before);
});

test('a restore that takes the lock itself holds it throughout and releases it afterwards', (t) => {
  const f = fixture(t);
  const lock = path.join(f.root, 'own.lock');
  const held = [];
  const copy = (from, to) => {
    held.push(fs.existsSync(path.join(lock, 'owner')));
    return copyVerified(from, to);
  };
  restore('example-start', restoreIn(f, { lock, stamp: 'took', copy }));
  assert.deepEqual(held, [true, true], 'the lock was held during both copies');
  assert.equal(fs.existsSync(lock), false, 'the lock is gone after the restore');
  assert.deepEqual(measure(f.folder), measure(f.snapshot));
});

test('the command line exits 1 when the save is as it was', (t) => {
  const f = fixture(t);
  fs.mkdirSync(path.join(f.snapshots, 'nothing'));
  const result = runSnapshot(['restore', 'nothing', '--config', f.toml, '--snapshots', f.snapshots, '--before', f.before, '--lock', path.join(f.root, 'a.lock')]);
  assert.equal(result.status, 1, result.stderr);
  assert.match(result.stderr, /is empty/);
});

// The failure paths of the swap and its clean-up. `ops` replaces rename, rm and rmdir; each wraps the real one.
const realOps = { rename: fs.renameSync, rm: fs.rmSync, rmdir: fs.rmdirSync };
const sibling = (f) => fs.readdirSync(path.dirname(f.folder)).sort();

test('a failed rename of the live folder leaves the save as it was and no staging folder', (t) => {
  const f = fixture(t);
  const before = manifest(f.folder);
  const ops = { rename: (a, b) => { if (a === f.folder) throw new Error('in use'); realOps.rename(a, b); } };
  assert.throws(() => restore('example-start', restoreIn(f, { ops, stamp: 's3a' })), (err) => /stopped before the save was touched: in use/.test(err.message) && err.touched === false);
  assert.deepEqual(manifest(f.folder), before);
  assert.deepEqual(sibling(f), [path.basename(f.folder)]);
});

test('a failed swap-in is rolled back, and a clean-up that fails afterwards does not turn it into a failed rollback', (t) => {
  const f = fixture(t);
  const before = manifest(f.folder);
  const staging = `${f.folder}.restore-s3b`;
  const ops = {
    rename: (a, b) => { if (a === staging) throw new Error('held by a scanner'); realOps.rename(a, b); },
    rm: (p, o) => { if (p === staging) throw new Error('still held'); realOps.rm(p, o); },
  };
  assert.throws(() => restore('example-start', restoreIn(f, { ops, stamp: 's3b' })), (err) => /was rolled back, the save is as it was/.test(err.message) && err.touched === false);
  assert.deepEqual(manifest(f.folder), before, 'the old save is back');
});

test('a folder something else made at the save path is kept aside, never deleted', (t) => {
  const f = fixture(t);
  const before = manifest(f.folder);
  const staging = `${f.folder}.restore-s3c`;
  const ops = {
    rename: (a, b) => {
      if (a === staging) {
        fs.mkdirSync(f.folder);
        fs.writeFileSync(path.join(f.folder, 'made-by-someone-else'), 'keep me');
        throw new Error('the path is taken');
      }
      realOps.rename(a, b);
    },
  };
  assert.throws(() => restore('example-start', restoreIn(f, { ops, stamp: 's3c' })), /rolled back.*kept in .*\.found-s3c/);
  assert.deepEqual(manifest(f.folder), before, 'the old save is back');
  assert.equal(fs.readFileSync(path.join(`${f.folder}.found-s3c`, 'made-by-someone-else'), 'utf8'), 'keep me');
});

test('a rollback that fails is reported as such, naming the intact old folder and the backup', (t) => {
  const f = fixture(t);
  const before = manifest(f.folder);
  const staging = `${f.folder}.restore-s3d`;
  const old = `${f.folder}.old-s3d`;
  const ops = {
    rename: (a, b) => {
      if (a === staging) throw new Error('swap failed');
      if (a === old) throw new Error('rollback failed');
      realOps.rename(a, b);
    },
  };
  assert.throws(
    () => restore('example-start', restoreIn(f, { ops, stamp: 's3d' })),
    (err) => err.touched === true && err.message.includes(old) && err.message.includes(path.join(f.before, 's3d-example-start')),
  );
  assert.deepEqual(manifest(old), before, 'the old folder holds the save');
  assert.deepEqual(manifest(path.join(f.before, 's3d-example-start')), before, 'and so does the backup');
});

test('a lock that cannot be released never replaces the outcome', (t) => {
  const f = fixture(t);
  const lock = path.join(f.root, 'stuck.lock');
  const ops = { rmdir: () => { throw new Error('busy'); } };
  const done = restore('example-start', restoreIn(f, { lock, ops, stamp: 'ok' }));
  assert.deepEqual(manifest(f.folder), manifest(f.snapshot), 'the restore succeeded');
  assert.equal(done.backup, path.join(f.before, 'ok-example-start'));
  const g = fixture(t);
  const staging = `${g.folder}.restore-bad`;
  const old = `${g.folder}.old-bad`;
  const bad = {
    rmdir: () => { throw new Error('busy'); },
    rename: (a, b) => { if (a === staging || a === old) throw new Error('no'); realOps.rename(a, b); },
  };
  assert.throws(() => restore('example-start', { ...restoreIn(g), lock: path.join(g.root, 'stuck.lock'), ops: bad, stamp: 'bad' }), (err) => err.touched === true);
});

test('an old folder that cannot be removed is reported, and the restore still stands', (t) => {
  const f = fixture(t);
  const old = `${f.folder}.old-s4`;
  const ops = { rm: (p, o) => { if (p === old) throw new Error('held'); realOps.rm(p, o); } };
  const done = restore('example-start', restoreIn(f, { ops, stamp: 's4' }));
  assert.equal(done.oldLeft, old);
  assert.deepEqual(manifest(f.folder), manifest(f.snapshot));
});

test('verification compares each file: same count and bytes with a changed file is caught and rolled back', (t) => {
  const f = fixture(t);
  const before = manifest(f.folder);
  let calls = 0;
  const copy = (from, to) => {
    calls += 1;
    const done = copyVerified(from, to);
    if (calls === 2) {
      const file = path.join(to, 'Player~fedcba9876543210', 'v3', '0');
      const text = fs.readFileSync(file, 'utf8');
      fs.writeFileSync(file, text.split('').reverse().join(''));
    }
    return done;
  };
  assert.throws(() => restore('example-start', restoreIn(f, { copy, stamp: 'same-size' })), /rolled back.*Player~fedcba9876543210\/v3\/0 /);
  assert.deepEqual(manifest(f.folder), before);
});

test('a backup that fails part-way is never left under a verified name', (t) => {
  const f = fixture(t);
  const copy = (from, to) => {
    fs.mkdirSync(to, { recursive: true });
    fs.writeFileSync(path.join(to, 'half'), 'x');
    throw new Error('the disk is full');
  };
  assert.throws(() => restore('example-start', restoreIn(f, { copy, stamp: 'part' })), /stopped before the save was touched/);
  assert.equal(fs.existsSync(path.join(f.before, 'part-example-start')), false);
  assert.equal(fs.existsSync(path.join(f.before, 'part-example-start.partial')), false);
});

test('a refused --before inside the save leaves no folder there', (t) => {
  const f = fixture(t);
  const before = manifest(f.folder);
  assert.throws(() => restore('example-start', restoreIn(f, { before: path.join(f.folder, 'backups', 'deeper') })), /overlap: refusing/);
  assert.deepEqual(manifest(f.folder), before, 'not even an empty folder was made');
});

test('a stamp is a plain name too, and device names are no names', (t) => {
  const f = fixture(t);
  for (const stamp of ['../x', 'a/b', '..', 'x.']) {
    assert.throws(() => restore('example-start', restoreIn(f, { stamp })), /is not a plain stamp/, stamp);
  }
  for (const name of ['CON', 'nul', 'com1', 'LPT9.txt', 'aux.save']) assert.throws(() => plainName(name), /not a plain/, name);
  assert.equal(plainName('console-start'), 'console-start');
});

test('a running game refuses a restore and a save-as', (t) => {
  const f = fixture(t);
  const before = manifest(f.folder);
  const running = (image) => image === 'victory.exe';
  assert.throws(() => restore('example-start', restoreIn(f, { isRunning: running })), /victory\.exe is running: refusing/);
  assert.throws(() => saveAs('during', { config: f.toml, snapshots: f.snapshots, isRunning: running }), /victory\.exe is running/);
  assert.deepEqual(manifest(f.folder), before);
});

test('leftovers of an interrupted restore refuse the next one, saying how to recover', (t) => {
  const f = fixture(t);
  fs.mkdirSync(`${f.folder}.old-crashed`);
  assert.throws(() => restore('example-start', restoreIn(f)), /A restore was interrupted here\. A <folder>\.old-<stamp> folder, if it is whole, holds the save from before it; a \.restore-, \.failed- or \.found- folder is not the save/);
});

test('the running-game check fails closed, on Windows and elsewhere', () => {
  const answer = (status, stdout = '', error) => () => ({ status, stdout, error });
  const win = { platform: 'win32' };
  assert.equal(processRunning('victory.exe', { ...win, run: answer(0, '"victory.exe","4242","Console","1","1,024 K"\r\n') }), true);
  assert.equal(processRunning('victory.exe', { ...win, run: answer(0, '"victory-tools.exe","1","Console","1","1 K"\r\n') }), false, 'an exact image name only');
  assert.equal(processRunning('victory.exe', { ...win, run: answer(0, 'INFO: No tasks are running which match the specified criteria.\r\n') }), false);
  assert.throws(() => processRunning('victory.exe', { ...win, run: answer(1, '') }), /cannot tell whether victory\.exe is running \(tasklist exit 1\): refusing/);
  assert.throws(() => processRunning('victory.exe', { ...win, run: answer(null, '', new Error('ENOENT')) }), /cannot tell .*ENOENT/);
  const nix = { platform: 'linux' };
  assert.equal(processRunning('victory.exe', { ...nix, run: answer(0) }), true);
  assert.equal(processRunning('victory.exe', { ...nix, run: answer(1) }), false);
  assert.throws(() => processRunning('victory.exe', { ...nix, run: answer(2) }), /pgrep exit 2/);
});

test('a game that starts during the copies stops the restore before the swap', (t) => {
  const f = fixture(t);
  const before = manifest(f.folder);
  let asked = 0;
  const isRunning = () => (asked += 1) > 1;
  assert.throws(() => restore('example-start', restoreIn(f, { isRunning, stamp: 'late' })), (err) => /stopped before the save was touched: victory\.exe started during the copies/.test(err.message) && err.touched === false);
  assert.equal(asked, 2);
  assert.deepEqual(manifest(f.folder), before);
  assert.deepEqual(sibling(f), [path.basename(f.folder)]);
});

// The runtime opens VICTORY_DATA_DIR when it is set, else the config file's data_dir (victory-core config.rs). A restore
// that followed only the file would replace one save while the game opened another.
test('the save folder is the one the runtime opens: VICTORY_DATA_DIR first, then the config file', (t) => {
  const f = fixture(t);
  // A second data tree, as a sandbox makes one, named only by VICTORY_DATA_DIR.
  const other = path.join(f.root, 'sandbox-data');
  const otherFolder = path.join(other, '0000ABCD', 'saves', 'feedfacefeedface', GUID);
  writeFile(path.join(otherFolder, 'box-two~0123456789abcdef', 'v1', '0'), 'sandbox-save');
  const fileTree = manifest(f.folder);
  assert.deepEqual(runtimeDataDir(f.toml, { VICTORY_DATA_DIR: other }), { dir: other, from: 'VICTORY_DATA_DIR' });
  assert.equal(runtimeDataDir(f.toml, { VICTORY_DATA_DIR: '  ' }).from, 'data_dir in the config file', 'empty is unset');
  assert.equal(runtimeDataDir(f.toml, {}).dir, f.data);
  const done = restore('example-start', restoreIn(f, { env: { VICTORY_DATA_DIR: other }, stamp: 'sandbox' }));
  assert.equal(done.dataFrom, 'VICTORY_DATA_DIR');
  assert.equal(done.folder, otherFolder);
  assert.deepEqual(manifest(otherFolder), manifest(f.snapshot), 'the folder the game would open got the snapshot');
  assert.deepEqual(manifest(f.folder), fileTree, "the config file's folder is untouched");
  const saved = saveAs('from-sandbox', { config: f.toml, snapshots: f.snapshots, env: { VICTORY_DATA_DIR: other }, isRunning: noGame });
  assert.equal(saved.dataFrom, 'VICTORY_DATA_DIR');
  assert.deepEqual(manifest(saved.snapshot), manifest(otherFolder));
});

test('the command line says which source named the data folder', (t) => {
  const f = fixture(t);
  const other = path.join(f.root, 'sandbox-data');
  writeFile(path.join(other, '0000ABCD', 'saves', 'feedfacefeedface', GUID, 'box-two~0123456789abcdef', 'v1', '0'), 'x');
  const args = ['restore', 'example-start', '--config', f.toml, '--snapshots', f.snapshots, '--before', f.before, '--lock', path.join(f.root, 'a.lock'), '--game-process', 'no-game-for-save-tests.exe'];
  const viaEnv = spawnSync(process.execPath, [script, ...args, '--stamp', 'one'], { encoding: 'utf8', env: isolatedEnv({ VICTORY_DATA_DIR: other }) });
  assert.equal(viaEnv.status, 0, viaEnv.stderr);
  assert.match(viaEnv.stdout, /the data folder from VICTORY_DATA_DIR/);
  const viaFile = spawnSync(process.execPath, [script, ...args, '--stamp', 'two'], { encoding: 'utf8', env: isolatedEnv({ VICTORY_DATA_DIR: '' }) });
  assert.equal(viaFile.status, 0, viaFile.stderr);
  assert.match(viaFile.stdout, /the data folder from data_dir in the config file/);
});
