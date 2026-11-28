import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { isolatedEnv } from './test-isolation.mjs';
import { readRoute, routeNames } from './route-keys.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'route-keys.mjs');

function runRoute(args) {
  return spawnSync(process.execPath, [script, ...args], { encoding: 'utf8' });
}

function tempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'victory-routes-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

test('a route can hold the settle by time as well as give its window', (t) => {
  const dir = tempDir(t);
  fs.writeFileSync(path.join(dir, 'held.keys'), ['# settled-seconds: 75', '# settled-after: 150', '55:Enter', ''].join('\n'));
  const route = readRoute(dir, 'held');
  assert.equal(route.settledSeconds, '75');
  assert.equal(route.settledAfter, '150');
  assert.deepEqual(route.keys, ['55:Enter']);
});

test('a route reads its comments, its header and its entries, joined with commas', (t) => {
  const dir = tempDir(t);
  fs.writeFileSync(
    path.join(dir, 'sample.keys'),
    [
      '# sample: a route for the test.',
      '#',
      '# heavy-seconds: 42',
      '# settled-seconds: 75',
      '',
      '55:Enter',
      '65:Enter',
      '  70:J:8  # a held turn',
      '# a trailing comment only',
      '80:Backspace:4',
      '',
    ].join('\n'),
  );
  assert.deepEqual(readRoute(dir, 'sample'), {
    keys: ['55:Enter', '65:Enter', '70:J:8', '80:Backspace:4'],
    heavySeconds: '42',
    settledSeconds: '75',
    settledAfter: null,
  });
  assert.deepEqual(routeNames(dir), ['sample']);
});

test('the command prints the keys, or the heavy-seconds header', (t) => {
  const dir = tempDir(t);
  fs.writeFileSync(path.join(dir, 'sample.keys'), '# heavy-seconds: 42\n# settled-seconds: 75\n55:Enter\n70:J:8\n');
  let result = runRoute(['keys', dir, 'sample']);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), '55:Enter,70:J:8');
  result = runRoute(['heavy-seconds', dir, 'sample']);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), '42');
  result = runRoute(['settled-seconds', dir, 'sample']);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), '75');
});

test('an unknown route fails and lists the routes that exist', (t) => {
  const dir = tempDir(t);
  fs.writeFileSync(path.join(dir, 'demo-turn.keys'), '55:Enter\n');
  fs.writeFileSync(path.join(dir, 'demo-skip.keys'), '55:Enter\n');
  const result = runRoute(['keys', dir, 'nowhere']);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /no route 'nowhere'; routes: demo-skip, demo-turn/);
});

test('a route with no keys fails rather than running an empty key list', (t) => {
  const dir = tempDir(t);
  fs.writeFileSync(path.join(dir, 'empty.keys'), '# heavy-seconds: 10\n# nothing else\n');
  const result = runRoute(['keys', dir, 'empty']);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /route 'empty' has no keys/);
});

test('the harness refuses --keys and --route together, and lists routes for a missing one', () => {
  // The script exits at argument parsing for both, before any binary or game check, so nothing runs.
  const harness = path.join(here, 'run-world.sh');
  let result = spawnSync('bash', [harness, '--keys', '55:Enter', '--route', 'demo-turn'], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 2, result.stderr);
  assert.match(result.stderr, /--keys and --route together/);
  result = spawnSync('bash', [harness, '--route', 'nowhere'], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(result.status, 2, result.stderr);
  assert.match(result.stderr, /no route 'nowhere'; routes: \S/);
});

// The team's protocol route: their 270 s test, the stand keep-alive and ten walk cycles, every entry one the
// scripted-key reader takes (a letter, a digit or a named key; a hold in whole seconds) and all of them after the
// settle ("+").
// The key routes follow places in the title, so the public tree does not carry them: there the test skips, and says so.
const NO_ROUTES = fs.existsSync(path.join(here, 'routes')) ? false : 'the key routes are not part of this tree';

test('team-270: the route holds the settle at 275 s and only keys the scripted reader takes', { skip: NO_ROUTES }, () => {
  const route = readRoute(path.join(here, 'routes'), 'team-270');
  assert.ok(route);
  // 275 s: 5 s past the protocol's last window, so PresentMon's last row still covers the walk to +270.
  assert.equal(route.settledSeconds, '275');
  const named = ['Enter', 'Esc', 'Tab', 'Back', 'Up', 'Down', 'Left', 'Right'];
  for (const entry of route.keys) {
    const m = entry.match(/^\+(\d+):([A-Za-z0-9]+)(?::(\d+))?$/);
    assert.ok(m, `entry not in the reader's form: ${entry}`);
    assert.ok(named.includes(m[2]) || /^[A-Za-z0-9]$/.test(m[2]), `key not taken by the reader: ${entry}`);
  }
  assert.deepEqual(
    route.keys.filter((entry) => entry.includes(':Down')),
    ['+30:Down', '+60:Down', '+90:Down', '+120:Down'],
  );
  assert.ok(route.keys.includes('+150:W:3'));
  assert.ok(route.keys.includes('+267:S:3'), 'the last walk cycle ends at +270');
  // Everything before +150 is the d-pad keep-alive, and no stick key is pressed before the walk phase.
  const before = route.keys.filter((entry) => Number(entry.match(/^\+(\d+)/)[1]) < 150);
  assert.deepEqual(before, ['+30:Down', '+60:Down', '+90:Down', '+120:Down']);
  assert.ok(before.every((entry) => !/:[WJLS]:/.test(entry)), 'no stick key before +150');
});
