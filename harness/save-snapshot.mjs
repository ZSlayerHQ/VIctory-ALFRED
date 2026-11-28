#!/usr/bin/env node
// The title's save folder as a repeatable fixture: restore a named snapshot before a run, or take one after it.
//
//   node harness/save-snapshot.mjs restore <name> --config <victory.toml> [--snapshots <dir>] [--before <dir>]
//        [--lock <path>] [--caller-holds-lock] [--allow-empty] [--stamp <text>] [--game-process <image>]
//   node harness/save-snapshot.mjs save-as <name> --config <victory.toml> [--snapshots <dir>] [--game-process <image>]
//
// The save folder is the title's config folder below the data folder the runtime opens: VICTORY_DATA_DIR when set,
// else the config file's data_dir (the runtime's own order): <data>/<title>/saves/<user>/<config>.
// Each level must hold exactly one folder, or the command refuses and says what it found.
//
// A snapshot (snapshots/<name>/, a plain name) holds the container folders as the runtime lays them out. restore
// copies the current folder to a backup and the snapshot to a staging folder beside the live one, writing both to
// disk and verifying every name, folder, size and SHA-256, before the live folder is touched; then it swaps the
// staging folder in by renames, and puts the old folder back if that fails (restore() says how). It refuses
// overlapping folders, an empty snapshot unless --allow-empty, leftovers of an interrupted restore, and a running
// game, and holds the run lock throughout. Exit 1: the save is as it was. Exit 3: a rollback failed; the old folder
// and the verified backup both hold the previous save. Exit 4: restored, but the old folder could not be removed.
// No file contents are ever printed. Known limits: each file is read whole to hash it, so one over 2 GiB makes the
// restore refuse; and each copied file is opened for writing to flush it, so a read-only file in the save or the
// snapshot makes it refuse. Both refusals leave the save as it was.
import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadLocalEnv } from './local-env.mjs';

// Snapshots live in <root>/snapshots and before-run backups in <root>/before-runs. VICTORY_SAVE_BACKUPS sets the root,
// from the environment or harness/local.env (local-env.mjs); the default is beside the user's home, which every
// machine has; --snapshots and --before override both. This PC's local.env names its own root.
export const DEFAULT_SAVE_BACKUPS = path.join(os.homedir(), 'victory-save-backups').split(String.fromCharCode(92)).join('/');

export function saveBackupsRoot(env = process.env) {
  const given = String(env.VICTORY_SAVE_BACKUPS ?? '').trim().split(String.fromCharCode(92)).join('/').replace(/\/+$/, '');
  return given || DEFAULT_SAVE_BACKUPS;
}

/// The `data_dir` of a victory.toml (single- or double-quoted; a double-quoted value unescapes backslashes).
export function dataDir(configPath) {
  const text = fs.readFileSync(configPath, 'utf8');
  const single = text.match(/^\s*data_dir\s*=\s*'([^']*)'/m);
  if (single) return single[1];
  const double = text.match(/^\s*data_dir\s*=\s*"((?:[^"\\]|\\.)*)"/m);
  if (double) return double[1].replace(/\\(.)/g, '$1');
  throw new Error(`no data_dir in ${configPath}`);
}

/// The one folder at `dir`, or an error naming what is there. `what` names the level in the message.
export function oneFolder(dir, what) {
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true }).filter((e) => e.isDirectory());
  } catch (err) {
    throw new Error(`cannot list ${what} ${dir}: ${err.message}`);
  }
  if (entries.length !== 1) {
    const names = entries.map((e) => e.name).join(', ') || 'none';
    const interrupted = entries.some((e) => /\.(restore|old|failed|found)-[^/]*$/.test(e.name))
      ? '. A restore was interrupted here. A <folder>.old-<stamp> folder, if it is whole, holds the save from before it; a .restore-, .failed- or .found- folder is not the save; and the backup under before-runs (the newest not ending in .partial) is the verified copy. Keep the folder the title should use under its own name, and move every other one out of this folder'
      : '';
    throw new Error(`expected exactly one ${what} under ${dir}, found ${entries.length}: ${names}${interrupted}`);
  }
  return path.join(dir, entries[0].name);
}

/// The title's save folder for a run config.
/// The data folder the runtime will open, by the runtime's own rule (victory-core config.rs): VICTORY_DATA_DIR when set
/// and not empty, else the config file's data_dir. `from` says which, for the log. A restore that read only the file
/// would replace the save in one folder while a game run with VICTORY_DATA_DIR opens another.
export function runtimeDataDir(configPath, env = process.env) {
  const given = String(env.VICTORY_DATA_DIR ?? '').trim();
  if (given) return { dir: given, from: 'VICTORY_DATA_DIR' };
  return { dir: dataDir(configPath), from: 'data_dir in the config file' };
}

export function saveFolder(configPath, env = process.env) {
  const data = runtimeDataDir(configPath, env).dir;
  const title = oneFolder(data, 'title folder');
  const saves = path.join(title, 'saves');
  const user = oneFolder(saves, 'user folder');
  return oneFolder(user, 'config folder');
}

/// A folder's file count and total bytes, recursively. For messages only: `manifest` is what verifies a copy.
export function measure(dir) {
  let files = 0;
  let bytes = 0;
  const walk = (at) => {
    for (const entry of fs.readdirSync(at, { withFileTypes: true })) {
      const full = path.join(at, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile()) {
        files += 1;
        bytes += fs.statSync(full).size;
      } else throw new Error(`not a plain file or folder: ${full}`);
    }
  };
  walk(dir);
  return { files, bytes };
}

const say = (m) => `${m.files} files, ${m.bytes} bytes`;

/// Every entry below `dir`, sorted: a folder as `<path>/`, a file as `<path> <size> <sha-256>`. Two folders with equal
/// manifests hold the same names, the same folders (empty ones too) and the same bytes in each file.
export function manifest(dir) {
  const out = [];
  const walk = (at, rel) => {
    for (const entry of fs.readdirSync(at, { withFileTypes: true })) {
      const full = path.join(at, entry.name);
      const name = rel ? `${rel}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        out.push(`${name}/`);
        walk(full, name);
      } else if (entry.isFile()) {
        const bytes = fs.readFileSync(full);
        out.push(`${name} ${bytes.length} ${crypto.createHash('sha256').update(bytes).digest('hex')}`);
      } else throw new Error(`not a plain file or folder: ${full}`);
    }
  };
  walk(dir, '');
  return out.sort();
}

const sameManifest = (a, b) => a.length === b.length && a.every((entry, i) => entry === b[i]);
const fileCount = (m) => m.filter((entry) => !entry.endsWith('/')).length;

/// The first name where two manifests differ, for a message (names only, never contents).
function firstDifference(want, got) {
  const has = new Set(got);
  const wanted = new Set(want);
  const name = (entry) => entry.split(' ')[0];
  const missing = want.find((entry) => !has.has(entry));
  if (missing) return `${name(missing)} is missing or differs`;
  const extra = got.find((entry) => !wanted.has(entry));
  return extra ? `${name(extra)} should not be there` : 'no difference';
}

/// Writes every file below `dir` to disk (fsync), and each folder where the system allows it (Windows does not).
export function flushTree(dir) {
  const sync = (p, flags) => {
    const fd = fs.openSync(p, flags);
    try {
      fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
  };
  const walk = (at) => {
    for (const entry of fs.readdirSync(at, { withFileTypes: true })) {
      const full = path.join(at, entry.name);
      if (entry.isDirectory()) walk(full);
      else sync(full, 'r+');
    }
    try {
      sync(at, 'r');
    } catch {
      // A folder cannot be opened for fsync on Windows; its files are on disk, which is what a restore relies on.
    }
  };
  walk(dir);
}

/// Copies `from`'s contents into `to` (made here), writes the copy to disk, and verifies it against the source: every
/// name, folder, size and SHA-256. Returns the copy's file count and bytes.
export function copyVerified(from, to) {
  fs.mkdirSync(to, { recursive: true });
  for (const entry of fs.readdirSync(from)) {
    fs.cpSync(path.join(from, entry), path.join(to, entry), { recursive: true });
  }
  flushTree(to);
  const source = manifest(from);
  const copy = manifest(to);
  if (!sameManifest(source, copy)) {
    throw new Error(`the copy of ${from} in ${to} does not match: ${firstDifference(source, copy)}`);
  }
  return measure(to);
}

// Windows' device names, which are no folder whatever their extension.
const DEVICE_NAME = /^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?$/i;

/// A snapshot name (or a stamp) is one plain folder name: letters, digits, '.', '_', '-'; never a path, never `.` or
/// `..`, never a device name, never ending in '.'.
export function plainName(name, what = 'snapshot name') {
  if (
    typeof name !== 'string' ||
    !/^[A-Za-z0-9._-]+$/.test(name) ||
    /^\.+$/.test(name) ||
    name.endsWith('.') ||
    DEVICE_NAME.test(name)
  ) {
    throw new Error(`'${name}' is not a plain ${what} (letters, digits, '.', '_', '-'; not a path or a device name)`);
  }
  return name;
}

/// The real path of `p`, with forward slashes: of `p` itself when it exists, else of its nearest existing parent with
/// the rest added, so a folder can be compared before it is made.
const real = (p) => {
  let at = path.resolve(p);
  const rest = [];
  while (!fs.existsSync(at)) {
    const up = path.dirname(at);
    if (up === at) break;
    rest.unshift(path.basename(at));
    at = up;
  }
  return path.join(fs.realpathSync.native(at), ...rest).split(path.sep).join('/');
};
const inside = (a, b) => a === b || a.startsWith(`${b.replace(/\/+$/, '')}/`);

/// Refuses unless no two of the named folders are the same or one inside another.
export function disjoint(folders) {
  const entries = Object.entries(folders);
  for (const [i, [an, a]] of entries.entries()) {
    for (const [bn, b] of entries.slice(i + 1)) {
      if (inside(a, b) || inside(b, a)) throw new Error(`the ${an} folder and the ${bn} folder overlap: refusing`);
    }
  }
}

/// Whether a process of exactly this image name is running (Windows' tasklist; elsewhere `pgrep -x`).
/// It fails closed: an answer it cannot read (tasklist failing, pgrep's error status) throws, and a restore refuses.
/// `run` and `platform` replace spawnSync and process.platform, for tests.
export function processRunning(image, { run = spawnSync, platform = process.platform } = {}) {
  const windows = platform === 'win32';
  const result = windows
    ? run('tasklist', ['/FI', `IMAGENAME eq ${image}`, '/NH', '/FO', 'CSV'], { encoding: 'utf8' })
    : run('pgrep', ['-x', image], { encoding: 'utf8' });
  const cannot = (why) => new Error(`cannot tell whether ${image} is running (${why}): refusing`);
  if (result.error) throw cannot(result.error.message);
  if (!windows) {
    if (result.status === 0) return true;
    if (result.status === 1) return false;
    throw cannot(`pgrep exit ${result.status}`);
  }
  if (result.status !== 0) throw cannot(`tasklist exit ${result.status}`);
  const want = `"${image.toLowerCase()}",`;
  return String(result.stdout).split(/\r?\n/).some((line) => line.toLowerCase().startsWith(want));
}

/// A restore that failed. `touched` says whether the live save folder may differ from how it was: false means it is
/// exactly as it was (nothing was changed, or a swap was rolled back); true means the rollback itself failed.
export class RestoreError extends Error {
  constructor(message, touched) {
    super(message);
    this.touched = touched;
  }
}

// Runs `step`, and reports instead of throwing: for clean-up whose failure changes nothing about the save.
function bestEffort(what, step) {
  try {
    step();
  } catch (err) {
    console.error(`save-snapshot: ${what} failed (${err.message}); nothing about the save depends on it`);
  }
}

/// The run lock: a restore holds it for itself (made here, removed after), or, when the run that started it holds it
/// already (`callerHoldsLock`, from world-run.sh), checks that the lock's owner file carries that run's token, so
/// another run's lock never passes for this one's. A lock that cannot be released is reported, and never replaces
/// the restore's own outcome.
function withLock(lock, options, ops, work) {
  if (options.callerHoldsLock) {
    const token = options.lockToken ?? process.env.WORLD_RUN_LOCK_TOKEN;
    let owner = '';
    try {
      owner = fs.readFileSync(path.join(lock, 'owner'), 'utf8');
    } catch {
      throw new RestoreError(`the run lock ${lock} is not held: refusing`, false);
    }
    if (!token || !owner.split(/\r?\n/).includes(`token ${token}`)) {
      throw new RestoreError(`the run lock ${lock} is held by another run: refusing`, false);
    }
    return work();
  }
  try {
    fs.mkdirSync(lock);
  } catch {
    throw new RestoreError(`refusing while ${lock} exists: a run holds the harness lock`, false);
  }
  let result;
  let failure = null;
  try {
    fs.writeFileSync(path.join(lock, 'owner'), `save-snapshot restore, pid ${process.pid}\n`);
    result = work();
  } catch (err) {
    failure = err;
  }
  try {
    ops.rm(path.join(lock, 'owner'), { force: true });
    ops.rmdir(lock);
  } catch (err) {
    console.error(`save-snapshot: the lock ${lock} could not be released (${err.message}); remove it once no run is going`);
  }
  if (failure) throw failure;
  return result;
}

/// Replaces the live save folder with a snapshot, never leaving half a save:
///   1. the live folder is copied to <backup>.partial, written to disk and verified, then renamed to <backup>;
///   2. the snapshot is copied to a staging folder beside the live one, written to disk and verified;
///   3. the live folder is renamed to <save>.old-<stamp> and the staging folder renamed into its place, then verified;
///   4. the old folder is removed.
/// "Verified" compares every name, folder, size and SHA-256. A failure before step 3 leaves the live folder as it
/// was. A failure in step 3 puts the old folder back; whatever stands at the save's path then is renamed aside
/// (<save>.failed-<stamp> for the swapped-in copy, <save>.found-<stamp> for a folder the restore did not make), never
/// deleted unless the restore made it. Only a failed rollback leaves the live folder changed (`RestoreError.touched`),
/// and its message names the intact old folder and the verified backup. Clean-up that fails after the save is safe is
/// reported, not thrown. Refuses: a name or stamp that is not plain, overlapping folders, an empty snapshot unless
/// `allowEmpty`, leftovers of an interrupted restore, and a running game (`gameProcess`, default victory.exe).
/// `ops` ({ rename, rm, rmdir }) and `copy` replace the file operations, for tests.
export function restore(name, options = {}) {
  plainName(name);
  const ops = { rename: fs.renameSync, rm: fs.rmSync, rmdir: fs.rmdirSync, ...options.ops };
  const snapshots = options.snapshots ?? `${saveBackupsRoot()}/snapshots`;
  const before = options.before ?? `${saveBackupsRoot()}/before-runs`;
  const lock =
    options.lock ?? process.env.VICTORY_RUN_LOCK ?? path.join(path.dirname(fileURLToPath(import.meta.url)), 'run.lock');
  const stamp = options.stamp ?? new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').replace('Z', '');
  plainName(stamp, 'stamp');
  const snapshot = path.join(snapshots, name);
  if (!fs.existsSync(snapshot) || !fs.statSync(snapshot).isDirectory()) {
    const names = fs.existsSync(snapshots)
      ? fs.readdirSync(snapshots).filter((n) => fs.statSync(path.join(snapshots, n)).isDirectory())
      : [];
    throw new RestoreError(`no snapshot '${name}'; snapshots: ${names.join(', ') || 'none'}`, false);
  }
  const env = options.env ?? process.env;
  const dataFrom = runtimeDataDir(options.config, env).from;
  const folder = saveFolder(options.config, env);
  disjoint({ save: real(folder), snapshot: real(snapshot), backups: real(before) });
  const want = manifest(snapshot);
  if (fileCount(want) === 0 && !options.allowEmpty) {
    throw new RestoreError(`the snapshot '${name}' is empty: refusing to replace the save with nothing (--allow-empty)`, false);
  }
  const image = options.gameProcess ?? 'victory.exe';
  const isRunning = options.isRunning ?? processRunning;
  if (isRunning(image)) {
    throw new RestoreError(`${image} is running: refusing to change the save under a running game`, false);
  }
  const backup = path.join(before, `${stamp}-${name}`);
  const partial = `${backup}.partial`;
  const staging = `${folder}.restore-${stamp}`;
  const old = `${folder}.old-${stamp}`;
  const failedCopy = `${folder}.failed-${stamp}`;
  const found = `${folder}.found-${stamp}`;
  for (const taken of [backup, partial, staging, old, failedCopy, found]) {
    if (fs.existsSync(taken)) throw new RestoreError(`${taken} already exists: refusing to overwrite it`, false);
  }
  const copyFn = options.copy ?? copyVerified;

  return withLock(lock, options, ops, () => {
    // 1 and 2: nothing of the live folder changes until both copies verify.
    let backedUp;
    try {
      fs.mkdirSync(before, { recursive: true });
      backedUp = copyFn(folder, partial);
      ops.rename(partial, backup);
      copyFn(snapshot, staging);
    } catch (err) {
      bestEffort('removing the staging folder', () => ops.rm(staging, { recursive: true, force: true }));
      bestEffort('removing the partial backup', () => ops.rm(partial, { recursive: true, force: true }));
      throw new RestoreError(`restore of '${name}' stopped before the save was touched: ${err.message}`, false);
    }
    // 3: swap by renames, after asking again whether a game has started during the copies.
    try {
      if (isRunning(image)) throw new Error(`${image} started during the copies`);
      ops.rename(folder, old);
    } catch (err) {
      bestEffort('removing the staging folder', () => ops.rm(staging, { recursive: true, force: true }));
      throw new RestoreError(`restore of '${name}' stopped before the save was touched: ${err.message}`, false);
    }
    let swapped = false;
    try {
      ops.rename(staging, folder);
      swapped = true;
      const got = manifest(folder);
      if (!sameManifest(want, got)) throw new Error(`the restore does not match the snapshot: ${firstDifference(want, got)}`);
    } catch (err) {
      // Put the old folder back. What stands at the save's path is renamed aside, never deleted here.
      let aside = null;
      try {
        if (fs.existsSync(folder)) {
          ops.rename(folder, swapped ? failedCopy : found);
          aside = swapped ? failedCopy : found;
        }
        ops.rename(old, folder);
      } catch (rollback) {
        const keptAside = aside === found ? `; a folder found at the save's path is in ${found}` : '';
        throw new RestoreError(
          `restore of '${name}' failed (${err.message}) and putting the old save back failed too (${rollback.message}): ` +
            `the save from before is intact in ${old}, and verified in ${backup}${keptAside}`,
          true,
        );
      }
      // The save is back. What follows is clean-up, and its failure does not change that.
      bestEffort('removing the staging folder', () => ops.rm(staging, { recursive: true, force: true }));
      if (swapped) bestEffort('removing the failed copy', () => ops.rm(failedCopy, { recursive: true, force: true }));
      const kept = aside === found ? `; a folder found at the save's path was kept in ${found}` : '';
      throw new RestoreError(`restore of '${name}' failed and was rolled back, the save is as it was: ${err.message}${kept}`, false);
    }
    // 4: the old folder is no longer needed: the backup holds it, written to disk and verified.
    let oldLeft = null;
    try {
      ops.rm(old, { recursive: true });
    } catch (err) {
      oldLeft = old;
      console.error(`save-snapshot: the old save folder could not be removed (${err.message}); it is ${old}`);
    }
    return { folder, dataFrom, backup, files: backedUp.files, bytes: backedUp.bytes, oldLeft };
  });
}

export function saveAs(name, options = {}) {
  const snapshots = options.snapshots ?? `${saveBackupsRoot()}/snapshots`;
  plainName(name);
  const snapshot = path.join(snapshots, name);
  if (fs.existsSync(snapshot)) throw new Error(`the snapshot '${name}' already exists: refusing to overwrite it`);
  const env = options.env ?? process.env;
  const dataFrom = runtimeDataDir(options.config, env).from;
  const folder = saveFolder(options.config, env);
  disjoint({ save: real(folder), snapshots: real(snapshots) });
  const image = options.gameProcess ?? 'victory.exe';
  if ((options.isRunning ?? processRunning)(image)) {
    throw new Error(`${image} is running: refusing to copy a save the game may be writing`);
  }
  fs.mkdirSync(snapshots, { recursive: true });
  const copied = copyVerified(folder, snapshot);
  return { folder, dataFrom, snapshot, files: copied.files, bytes: copied.bytes };
}

function usage() {
  console.error(
    'usage: save-snapshot.mjs restore <name> --config <victory.toml> [--snapshots <dir>] [--before <dir>] [--lock <path>] [--caller-holds-lock] [--allow-empty] [--stamp <text>] [--game-process <image>]',
  );
  console.error('       save-snapshot.mjs save-as <name> --config <victory.toml> [--snapshots <dir>] [--game-process <image>]');
}

async function main(argv) {
  loadLocalEnv();
  const [what, name, ...rest] = argv;
  const options = {};
  for (let i = 0; i < rest.length; i += 1) {
    const arg = rest[i];
    if (arg === '--config') options.config = rest[++i];
    else if (arg === '--snapshots') options.snapshots = rest[++i];
    else if (arg === '--before') options.before = rest[++i];
    else if (arg === '--lock') options.lock = rest[++i];
    else if (arg === '--caller-holds-lock') options.callerHoldsLock = true;
    else if (arg === '--allow-empty') options.allowEmpty = true;
    else if (arg === '--game-process') options.gameProcess = rest[++i];
    else if (arg === '--stamp') options.stamp = rest[++i];
    else {
      console.error(`save-snapshot: unknown argument ${arg}`);
      usage();
      return 2;
    }
  }
  if (!['restore', 'save-as'].includes(what) || !name || !options.config) {
    usage();
    return 2;
  }
  try {
    if (what === 'restore') {
      const done = restore(name, options);
      console.log(`restored '${name}' into ${done.folder} (the data folder from ${done.dataFrom}; ${say({ files: done.files, bytes: done.bytes })} backed up to ${done.backup})`);
      if (done.oldLeft) return 4;
    } else {
      const done = saveAs(name, options);
      console.log(`saved '${name}' from ${done.folder} (the data folder from ${done.dataFrom}) into ${done.snapshot} (${say({ files: done.files, bytes: done.bytes })})`);
    }
    return 0;
  } catch (err) {
    console.error(`save-snapshot: ${err.message}`);
    // 3: the live save may have changed (a rollback failed); 1: it is as it was.
    return err instanceof RestoreError && err.touched ? 3 : 1;
  }
}

const isMain =
  process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  main(process.argv.slice(2))
    .then((code) => process.exit(code))
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}
