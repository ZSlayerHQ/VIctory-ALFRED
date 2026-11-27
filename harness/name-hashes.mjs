#!/usr/bin/env node
// After each shader hash the table knows, write the shader's name: a log, a summary or a worklist reads as words.
// usage: node name-hashes.mjs <file> [--names harness/logs/shader-names.tsv]
// The table is committed only in this private repo (D-034); the default here is the copy the names tool writes
// under harness/logs, and VICTORY_SHADER_NAMES overrides it (tests, or a table kept elsewhere).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

// A 16-hex hash that is not part of a longer run and has no name written after it yet.
const HASH_RE = /(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])(?! \()/g;

/// The table's hash-to-name map, or an empty map when the file is absent.
export function loadNames(file) {
  if (!file || !fs.existsSync(file)) return new Map();
  const names = new Map();
  for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    if (!line) continue;
    const cells = line.split('\t');
    if (cells.length < 5) continue;
    const hash = cells[0].trim();
    const name = cells[cells.length - 1].trim();
    if (/^[0-9a-f]{16}$/.test(hash) && name) names.set(hash, name);
  }
  return names;
}

/// `text` with ` (<name>)` after every hash `names` holds.
export function annotate(text, names) {
  return text.replace(HASH_RE, (hash) => (names.has(hash) ? `${hash} (${names.get(hash)})` : hash));
}

/// The table's path: `--names`, else VICTORY_SHADER_NAMES, else harness/logs/shader-names.tsv.
export function defaultNamesFile() {
  return process.env.VICTORY_SHADER_NAMES || path.join(here, 'logs', 'shader-names.tsv');
}

function main() {
  const args = process.argv.slice(2);
  let namesFile = defaultNamesFile();
  const at = args.indexOf('--names');
  if (at >= 0) {
    namesFile = args[at + 1];
    args.splice(at, 2);
  }
  const file = args[0];
  if (!file) {
    console.error('usage: node name-hashes.mjs <file> [--names harness/logs/shader-names.tsv]');
    process.exit(2);
  }
  const names = loadNames(namesFile);
  process.stdout.write(annotate(fs.readFileSync(file, 'utf8'), names));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
