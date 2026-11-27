// The frame line's cost part under both labels: "on the GPU" (builds before job/frame-timers) and "host waits (...)"
// (builds from it on, where the figure is named for what it measures). The capture groups keep their positions.
import test from 'node:test';
import assert from 'node:assert/strict';
import { COST_RE, parseLogLines } from './parse-run.mjs';

const head = '12.5 preparing: copies 3.1 ms 4.2 MB, textures 5.3 ms 6.4 MB; ';
const tail = '; 8.6 copying back 9.7 MB';
const want = ['12.5', '3.1', '4.2', '5.3', '6.4', '7.5', '8.6', '9.7'];

test('the cost part parses with the old label', () => {
  const m = COST_RE.exec(`${head}7.5 on the GPU${tail}`);
  assert.ok(m);
  assert.deepEqual(m.slice(1), want);
});

test('the cost part parses with the host-waits label, into the same groups', () => {
  const waits = "7.5 host waits (for the relay 1.5, for flights' fence 2.0, synchronous submissions sent to retired 4.0)";
  const m = COST_RE.exec(`${head}${waits}${tail}`);
  assert.ok(m);
  assert.deepEqual(m.slice(1), want);
});

test('the cost part still parses when the copied-back bytes are followed by more text', () => {
  const m = COST_RE.exec(`${head}7.5 on the GPU${tail} merged, 3.0 MB fetched`);
  assert.ok(m);
  assert.deepEqual(m.slice(1), want);
});

// The figure is the host's waits under either label (the audit of 2026-11-26: "GPU" was never card time), so a
// parsed frame names it hostWaitsMs and carries no field that calls it the GPU's.
test('a parsed frame names the figure hostWaitsMs under both labels, and no GPU field remains', () => {
  const frameLine = (cost) =>
    `2026-11-26T05:02:11.250000Z  INFO victory_gpu::host: frame 10, 27.4 fps, 36.5 ms/frame (${head}${cost}${tail}), ` +
    '5000 draws/frame, dropped 3 without a host pipeline (1 being made) and 7 indirect';
  const waits = "7.5 host waits (for the relay 1.5, for flights' fence 2.0, synchronous submissions sent to retired 4.0)";
  for (const cost of ['7.5 on the GPU', waits]) {
    const { summary } = parseLogLines([frameLine(cost)], 'run.log');
    assert.equal(summary.lastFrameLine.hostWaitsMs, 7.5, cost);
    assert.equal(summary.costs.hostWaitsMs, 7.5, cost);
    assert.equal(summary.worstFrames[0].hostWaitsMs, 7.5, cost);
    for (const record of [summary.lastFrameLine, summary.costs]) {
      assert.deepEqual(Object.keys(record).filter((key) => /gpu/i.test(key)), [], cost);
    }
  }
});

// settled-watch writes "waiting ?" when it has no texture count (the validator's latent finding of 2026-11-26); the
// readers of its marker must take both forms from the writer itself.
test('the settle marker regex reads both forms settled-watch writes', async () => {
  const { HARNESS_SETTLED_RE } = await import('./parse-run.mjs');
  const { settledMarker } = await import('./settled-watch.mjs');
  for (const marker of [settledMarker(75, 40, 2.4), settledMarker(12, null, 4.1, 'x')]) {
    assert.ok(HARNESS_SETTLED_RE.test(marker), marker);
  }
});
