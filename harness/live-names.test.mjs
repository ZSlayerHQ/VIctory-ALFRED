import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { LIVE_KINDS, LIVE_NAMES, normalizeLiveValue } from './live-names.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));

/// The runtime's live.rs: VICTORY_REPO, else the victory checkout beside harness/ (the outer repository) or the
/// outer repository's from a lane worktree (worktrees/<lane>/harness). Null where there is none (the public tree).
function runtimeLiveRs() {
  const roots = [process.env.VICTORY_REPO, path.join(here, '..', 'victory'), path.join(here, '..', '..', '..', 'victory')];
  for (const root of roots) {
    if (!root) continue;
    const file = path.join(root, 'crates', 'victory-gpu', 'src', 'live.rs');
    if (fs.existsSync(file)) return file;
  }
  return null;
}

const KIND = { OnOff: 'onoff', Integer: 'integer', List: 'list', Keys: 'keys', Mark: 'mark' };

test("the harness's live names and kinds equal the runtime's switches! table", (t) => {
  const file = runtimeLiveRs();
  if (!file) {
    t.skip('no victory checkout (set VICTORY_REPO): the runtime table cannot be read here');
    console.log('live-names: skipped, no victory checkout (set VICTORY_REPO)');
    return;
  }
  const text = fs.readFileSync(file, 'utf8');
  const table = text.slice(text.indexOf('switches! {'));
  const runtime = {};
  for (const match of table.matchAll(/^\s+[A-Z0-9_]+ => "([A-Z0-9_]+)", (OnOff|Integer|List|Keys|Mark),/gm)) {
    runtime[match[1]] = KIND[match[2]];
  }
  assert.ok(Object.keys(runtime).length >= 10, 'the table was found and read');
  assert.deepEqual([...LIVE_NAMES].sort(), Object.keys(runtime).sort(), 'the same names');
  assert.deepEqual(
    Object.fromEntries(Object.entries(LIVE_KINDS).sort()),
    Object.fromEntries(Object.entries(runtime).sort()),
    'the same kinds',
  );
});

test('a value is normalized as the runtime holds it', () => {
  assert.equal(normalizeLiveValue('VICTORY_FETCH_QUEUE', '1'), 'on');
  assert.equal(normalizeLiveValue('VICTORY_FETCH_QUEUE', ' No '), 'off');
  assert.equal(normalizeLiveValue('VICTORY_FETCH_QUEUE', ''), 'off');
  assert.equal(normalizeLiveValue('VICTORY_TEXTURE_LOW', '150'), '100');
  assert.equal(normalizeLiveValue('VICTORY_TEXTURE_LOW', '0'), '1');
  assert.equal(normalizeLiveValue('VICTORY_TEXTURE_LOW', 'abc'), null, 'refused: the runtime keeps the old value');
  assert.equal(normalizeLiveValue('LIVE_SKIP_DRAW', ' 12, ,34 '), '12,34');
});

test('a mark is a whole number, zero or more, as the runtime takes it; anything else keeps the old value', () => {
  assert.equal(normalizeLiveValue('LIVE_MARK', ' 7 '), '7');
  assert.equal(normalizeLiveValue('LIVE_MARK', '0'), '0');
  assert.equal(normalizeLiveValue('LIVE_MARK', '+12'), '12');
  assert.equal(normalizeLiveValue('LIVE_MARK', '-1'), null, 'below zero is refused');
  assert.equal(normalizeLiveValue('LIVE_MARK', 'next'), null);
  assert.equal(normalizeLiveValue('LIVE_MARK', ''), null, 'an empty mark is not a number');
});
