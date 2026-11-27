import assert from 'node:assert/strict';
import test from 'node:test';

import { dropWebAddresses, hasWebAddress, withheldText, withoutWebAddresses } from './web-address.mjs';

// The query mark of a web address, joined in at run time so that no address literal here carries one.
const Q = '?';

// Made-up addresses on reserved names (example.com, example.org): no real one is written here.
test('a line holding a web address is found: with a query, inside a longer line, in upper case, by www', () => {
  assert.equal(hasWebAddress('see https://example.com/a' + Q + 'b=1&c=2 for more'), true, 'an address with a query');
  assert.equal(hasWebAddress('2026-11-26T10:00:00Z WARN victory_gpu::x: fetched(https://example.org/x),retrying'), true, 'inside a longer line, between brackets');
  assert.equal(hasWebAddress('HTTP://EXAMPLE.COM/'), true, 'upper case scheme');
  assert.equal(hasWebAddress('Https://example.com'), true);
  assert.equal(hasWebAddress('opened file:///tmp/x'), true, 'any scheme of two letters or more');
  assert.equal(hasWebAddress('mirror at www.example.com today'), true, 'a www host');
  assert.equal(hasWebAddress('a,https://example.com;b'), true, 'after a comma, up to a semicolon');
});

test('a colon and slashes in a path are not an address', () => {
  for (const line of [
    'cache at D:/captures/run/stats.json',
    'E://victory-captures/x',
    'C:\\Users\\someone\\x',
    '\\\\server\\share\\x',
    'harness: capture: E:\\victory-captures\\20261126-1 (ratio 1:2)',
    'time 10:00:00.000Z frames 1..6',
    'a/b:/c',
  ]) {
    assert.equal(hasWebAddress(line), false, line);
  }
});

test('lines with an address are withheld and counted; a JSON value loses only its address strings', () => {
  const { lines, withheld } = withoutWebAddresses(['one', 'see https://example.com', 'two', 'WWW.example.org/x']);
  assert.deepEqual(lines, ['one', 'two']);
  assert.equal(withheld, 2);
  assert.equal(withheldText(1), '1 line withheld (they hold a web address)');
  assert.equal(withheldText(2), '2 lines withheld (they hold a web address)');
  const dropped = dropWebAddresses({ failures: ['a', 'b https://example.com'], ending: { text: 'http://example.com' }, n: 3 });
  assert.deepEqual(dropped.value, { failures: ['a'], ending: { text: null }, n: 3 });
  assert.equal(dropped.withheld, 2);
});
