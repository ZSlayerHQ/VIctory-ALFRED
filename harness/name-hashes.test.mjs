import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'name-hashes.mjs');

const KNOWN = '0123456789abcdef';
const OTHER = 'fedcba9876543210';
const UNKNOWN = 'aaaaaaaaaaaaaaaa';

function tempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'name-hashes-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

function table(dir) {
  const file = path.join(dir, 'shader-names.tsv');
  fs.writeFileSync(
    file,
    [
      `${KNOWN}\tpixel\tfx1\tdraw pass 1\tPS_Lighting_Wrapped`,
      `${OTHER}\tcompute\tfx2\tdispatch\tCS_Tiles_Wrapped`,
      '',
    ].join('\n'),
  );
  return file;
}

test('each known hash gains its name; unknown hashes and longer runs stay bare', (t) => {
  const dir = tempDir(t);
  const names = table(dir);
  const text = path.join(dir, 'worklist.txt');
  fs.writeFileSync(
    text,
    [
      `compute ${KNOWN} dropped`,
      `hash ${OTHER} here`,
      `unknown ${UNKNOWN} stays`,
      `longer 0${KNOWN}0 is not a hash`,
      `already ${KNOWN} (Existing) is left alone`,
      '',
    ].join('\n'),
  );
  const result = spawnSync(process.execPath, [script, text, '--names', names], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, new RegExp(`${KNOWN} \\(PS_Lighting_Wrapped\\)`));
  assert.match(result.stdout, new RegExp(`${OTHER} \\(CS_Tiles_Wrapped\\)`));
  assert.ok(result.stdout.includes(UNKNOWN), result.stdout);
  assert.ok(result.stdout.includes(`0${KNOWN}0`), 'a longer hex run is not annotated');
  assert.ok(result.stdout.includes(`${KNOWN} (Existing)`), 'an existing name is not doubled');
});

test('a missing table leaves the text unchanged', (t) => {
  const dir = tempDir(t);
  const text = path.join(dir, 'log.txt');
  fs.writeFileSync(text, `hash ${KNOWN}\n`);
  const result = spawnSync(process.execPath, [script, text, '--names', path.join(dir, 'none.tsv')], {
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout, `hash ${KNOWN}\n`);
});

test('summarize-run annotates the hashes in its output when the table exists', (t) => {
  const dir = tempDir(t);
  const names = table(dir);
  const log = path.join(dir, '20261124-230000-names.log');
  fs.writeFileSync(
    log,
    [
      'harness build: abc profile=diag console=series-x',
      `2026-11-24T12:00:00.000000Z  WARN victory_gpu::pipeline::host: host pipeline refused error=bad shaders=[compute ${KNOWN}]`,
      'harness: victory exit=0',
      '',
    ].join('\n'),
  );
  const result = spawnSync(process.execPath, [path.join(here, 'summarize-run.mjs'), log], {
    encoding: 'utf8',
    env: { ...process.env, VICTORY_SHADER_NAMES: names },
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, new RegExp(`${KNOWN} \\(PS_Lighting_Wrapped\\)`));
  // Without the table the summary is unchanged.
  const plain = spawnSync(process.execPath, [path.join(here, 'summarize-run.mjs'), log], {
    encoding: 'utf8',
    env: { ...process.env, VICTORY_SHADER_NAMES: path.join(dir, 'none.tsv') },
  });
  assert.equal(plain.status, 0, plain.stderr);
  assert.ok(!plain.stdout.includes('PS_Lighting_Wrapped'), plain.stdout);
});
