// stats-recorder.mjs on stats files written here (made-up objects in the runtime's two shapes; no real stats file).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { initialState, LOAD_END_RE, recordOnce, watch } from './stats-recorder.mjs';
import { readStatsHistory } from './presentmon-compare.mjs';

function files(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'stats-recorder-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return { dir, stats: path.join(dir, 'stats.json'), history: path.join(dir, 'stats.jsonl'), log: path.join(dir, 'run.log') };
}

// An object of the interval shape (job/timing-truth on): `seq` names it.
const interval = (seq, frame) => ({
  seq,
  t_ms: seq * 1000,
  unix_ms: 1795000000000 + seq * 1000,
  interval: { first_frame: frame - 3, last_frame: frame, frames: 4, wall_ms: 1000 },
  frame,
  frame_ms: 250,
  frame_ms_list: [250, 250, 250, 250],
  frame_ms_list_missing: 0,
});

test('each object is appended once, in order: a repeat is skipped, a torn file is read again at the next poll', (t) => {
  const f = files(t);
  let state = initialState();
  const write = (object) => fs.writeFileSync(f.stats, JSON.stringify(object, null, 2));
  write(interval(1, 4));
  state = recordOnce(state, f.stats, f.history);
  state = recordOnce(state, f.stats, f.history); // the same object read again
  fs.writeFileSync(f.stats, '{"seq": 2, "fra'); // mid-write
  state = recordOnce(state, f.stats, f.history);
  write(interval(2, 8));
  state = recordOnce(state, f.stats, f.history);
  fs.rmSync(f.stats); // between the runtime's writes: nothing there
  state = recordOnce(state, f.stats, f.history);
  write(interval(3, 12));
  state = recordOnce(state, f.stats, f.history);
  const text = fs.readFileSync(f.history, 'utf8');
  assert.equal(text.split('\n').filter(Boolean).length, 3, 'one line per object');
  assert.deepEqual(readStatsHistory(text).map((o) => o.seq), [1, 2, 3]);
  assert.equal(state.appended, 3);
  assert.equal(state.missed, 0);
});

test('a seq jump is counted as objects missed; the older shape, with no seq, is new when its frame moves', (t) => {
  const f = files(t);
  let state = initialState();
  for (const seq of [1, 4]) {
    fs.writeFileSync(f.stats, JSON.stringify(interval(seq, seq * 4)));
    state = recordOnce(state, f.stats, f.history);
  }
  assert.equal(state.missed, 2);
  const old = (frame) => JSON.stringify({ frame, fps: 4, frame_ms: 250, parts: {}, dispatches: 10 });
  for (const frame of [100, 100, 130]) {
    fs.writeFileSync(f.stats, old(frame));
    state = recordOnce(state, f.stats, f.history);
  }
  assert.equal(readStatsHistory(fs.readFileSync(f.history, 'utf8')).length, 4);
});

test("the recorder ends after run-world.sh's load-end line, with one last read of the stats file", async (t) => {
  const f = files(t);
  fs.writeFileSync(f.log, 'harness build: abc1234\n');
  fs.writeFileSync(f.stats, JSON.stringify(interval(1, 4)));
  const done = watch(f.stats, f.history, { log: f.log, pollMs: 20 });
  await new Promise((resolve) => setTimeout(resolve, 80));
  // The game's last object and its exit, written together: the last read must still take the object.
  fs.writeFileSync(f.stats, JSON.stringify(interval(2, 8)));
  fs.appendFileSync(f.log, 'harness: load at end: cpu 7% over 5 s\nharness: load end exit=0\n');
  assert.equal(await done, 0);
  assert.deepEqual(readStatsHistory(fs.readFileSync(f.history, 'utf8')).map((o) => o.seq), [1, 2]);
  assert.ok(LOAD_END_RE.test('harness: load end exit=0'));
});
