#!/usr/bin/env node
// Makes a save snapshot that also carries the title's profile settings, so a run restored from it does not meet the
// title's first-start screens (legal notices, terms) and loads its save unattended.
//
//   node harness/save-with-settings.mjs <snapshot> <new snapshot> --kind <container kind> --config <victory.toml>
//                                        [--snapshots <dir>]
//
// A snapshot restored by save-snapshot.mjs replaces the whole save folder. One that holds only a save container
// leaves the title without its settings container, and the title then asks for its first-time setup on every boot.
// This copies <snapshot> to <new snapshot> and adds the settings container the title has written in the LIVE save
// folder (run the title through its first-start screens once, then run this). You name that container's kind with
// --kind: the part of its folder name before the `~`, as your own save folder shows it; the tool treats it as opaque. It never overwrites a snapshot, never
// changes the live folder, and prints no path and no file contents: container names are printed up to their `~`.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { saveBackupsRoot, saveFolder } from './save-snapshot.mjs';

/// The container folders in `dir` whose name (up to `~`) matches `kind`.
export function containersOf(dir, kind) {
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.split('~')[0] === kind)
    .map((entry) => entry.name);
}

function copyTree(from, to) {
  fs.mkdirSync(to, { recursive: true });
  let files = 0;
  for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
    const [a, b] = [path.join(from, entry.name), path.join(to, entry.name)];
    if (entry.isDirectory()) files += copyTree(a, b);
    else {
      fs.copyFileSync(a, b, fs.constants.COPYFILE_EXCL);
      files += 1;
    }
  }
  return files;
}

/// Copies snapshot `from` to `to` under `snapshots`, then adds every `kind` container of `live`. Returns what it did.
export function withSettings({ snapshots, from, to, live, kind }) {
  if (!kind) throw new Error('name the kind of the settings container with --kind (the part of its folder name before the ~)');
  const source = path.join(snapshots, from);
  const target = path.join(snapshots, to);
  if (!fs.existsSync(source)) throw new Error(`snapshot ${from} does not exist`);
  if (fs.existsSync(target)) throw new Error(`snapshot ${to} exists already: a snapshot is never overwritten`);
  const settings = containersOf(live, kind);
  if (settings.length !== 1) throw new Error(`the live save folder holds ${settings.length} ${kind} containers, not one`);
  if (containersOf(source, kind).length) throw new Error(`snapshot ${from} already carries a ${kind} container`);
  const copied = copyTree(source, target);
  const added = copyTree(path.join(live, settings[0]), path.join(target, settings[0]));
  return { copied, added, kinds: fs.readdirSync(target).map((name) => name.split('~')[0]).sort() };
}

function main(argv) {
  const names = argv.filter((arg, i) => !arg.startsWith('--') && !['--config', '--snapshots', '--kind'].includes(argv[i - 1]));
  const value = (flag) => (argv.includes(flag) ? argv[argv.indexOf(flag) + 1] : undefined);
  if (names.length !== 2 || !value('--config')) throw new Error('usage: save-with-settings.mjs <snapshot> <new snapshot> --kind <container kind> --config <victory.toml> [--snapshots <dir>]');
  const result = withSettings({
    snapshots: value('--snapshots') ?? path.join(saveBackupsRoot(), 'snapshots'),
    from: names[0],
    to: names[1],
    live: saveFolder(value('--config')),
    kind: value('--kind'),
  });
  console.log(`snapshot ${names[1]}: ${result.copied} file(s) from ${names[0]}, ${result.added} settings file(s) from the live folder; holds: ${result.kinds.join(', ')}`);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(`save-with-settings: ${String(error.message).replace(/[A-Za-z]:[\\/][^\s'"]*/g, '<path>')}`);
    process.exit(1);
  }
}
