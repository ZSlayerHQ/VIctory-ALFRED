import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { endSelected } from './end-run.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));

test('the game is ended by what a fresh selection returns now, and nothing else', () => {
  const ended = [];
  const end = (pid) => {
    ended.push(pid);
    return true;
  };
  assert.deepEqual(endSelected(() => [42, 43], end), { outcome: 'ended', fresh: [42, 43], ended: [42, 43] });
  assert.deepEqual(ended, [42, 43]);
  ended.length = 0;
  assert.deepEqual(endSelected(() => [], end), { outcome: 'gone', fresh: [], ended: [] }, 'an empty selection: the game ended by itself');
  assert.deepEqual(endSelected(() => null, end), { outcome: 'unknown', fresh: null, ended: [] }, 'could not tell: nothing is ended');
  assert.deepEqual(ended, [], 'neither ends anything');
  assert.equal(endSelected(() => [7], () => false).outcome, 'failed');
});

test('every watcher that can end a run ends it through endSelected, never through ids it saw earlier', () => {
  for (const name of ['settled-watch.mjs', 'watchdog.mjs', 'heavy-watch.mjs', 'disk-guard.mjs']) {
    const text = fs.readFileSync(path.join(here, name), 'utf8');
    assert.match(text, /endSelected\(\(\) => selectionPids\(psFile\), endPid\)/, `${name} ends through a fresh selection`);
    assert.doesNotMatch(text, /alive\.filter\(endPid\)/, `${name} no longer ends the cached ids`);
  }
});
