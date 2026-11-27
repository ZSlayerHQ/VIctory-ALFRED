import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { containersOf, withSettings } from './save-with-settings.mjs';

function scratch(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'save-with-settings-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

function put(root, rel, text) {
  const file = path.join(root, rel);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text);
}

function fixture(t) {
  const dir = scratch(t);
  const snapshots = path.join(dir, 'snapshots');
  const live = path.join(dir, 'live');
  put(snapshots, 'house/box-a~aaaa/v1/0', 'save');
  put(snapshots, 'house/box-a~aaaa/v1/manifest', 'm');
  put(live, 'box-a~aaaa/v1/0', 'a later save');
  put(live, 'box-b~bbbb/v2/0', 'an autosave');
  put(live, 'prefs~cccc/v1/0', 'settings');
  put(live, 'prefs~cccc/v1/manifest', 'm');
  return { snapshots, live };
}

test('the new snapshot holds the old one and the live settings container, and nothing else of the live folder', (t) => {
  const { snapshots, live } = fixture(t);
  const result = withSettings({ snapshots, from: 'house', to: 'house-s', live, kind: 'prefs' });
  assert.deepEqual(result, { copied: 2, added: 2, kinds: ['box-a', 'prefs'] });
  assert.equal(fs.readFileSync(path.join(snapshots, 'house-s/box-a~aaaa/v1/0'), 'utf8'), 'save', "the snapshot's save, not the live one");
  assert.equal(fs.readFileSync(path.join(snapshots, 'house-s/prefs~cccc/v1/0'), 'utf8'), 'settings');
  assert.deepEqual(containersOf(path.join(snapshots, 'house-s'), 'box-b'), [], 'the autosave stays out');
  assert.deepEqual(fs.readdirSync(path.join(snapshots, 'house')), ['box-a~aaaa'], 'the old snapshot is untouched');
  assert.equal(fs.readFileSync(path.join(live, 'box-a~aaaa/v1/0'), 'utf8'), 'a later save', 'the live folder is untouched');
});

test('a snapshot is never overwritten, and a missing or doubled settings container is refused before anything is copied', (t) => {
  const { snapshots, live } = fixture(t);
  put(snapshots, 'taken/x', 'x');
  assert.throws(() => withSettings({ snapshots, from: 'house', to: 'taken', live, kind: 'prefs' }), /exists already/);
  assert.throws(() => withSettings({ snapshots, from: 'house', to: 'other', live }), /--kind/);
  assert.throws(() => withSettings({ snapshots, from: 'absent', to: 'new', live, kind: 'prefs' }), /does not exist/);
  put(live, 'prefs~dddd/v1/0', 'second');
  assert.throws(() => withSettings({ snapshots, from: 'house', to: 'new', live, kind: 'prefs' }), /holds 2 prefs containers/);
  assert.equal(fs.existsSync(path.join(snapshots, 'new')), false);
  fs.rmSync(path.join(live, 'prefs~dddd'), { recursive: true });
  fs.rmSync(path.join(live, 'prefs~cccc'), { recursive: true });
  assert.throws(() => withSettings({ snapshots, from: 'house', to: 'new', live, kind: 'prefs' }), /holds 0 prefs containers/);
});
