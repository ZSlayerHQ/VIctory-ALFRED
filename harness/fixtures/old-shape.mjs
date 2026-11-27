// The pre-rebuild shapes, kept for reading old logs and stats files: what the runtime wrote before job/timing-truth
// (world b1714baf6). The live stats file then held one second's sample about once in six, with no seq, clock or
// window; the segment line counted every present. Hand-written, made-up numbers in that shape; the shapes the
// runtime writes now are runtime-stats.mjs and runtime-lines.mjs, copied from its own tests.

/// The older one-second sample (made-up numbers): `parts` for the summed figures, `dispatches` a count since the
/// previous object, `stores_failed` and `pipelines_created` counts since the start, and `segment` the closed segments.
export function sampleObject(overrides = {}) {
  return {
    frame: 2048,
    fps: 4.0,
    frame_ms: 250.0,
    parts: { preparing_ms: 1.2, host_wait_ms: 6.0 },
    draws: 1500.0,
    dispatches: 1860,
    dropped: 3.0,
    stores_failed: 0,
    stores_failed_bytes: 0,
    pipelines_created: 900,
    pipelines_per_minute: 1.0,
    vram_mb: 18000,
    live: { VICTORY_FETCH_QUEUE: 'off' },
    segment: 1,
    ...overrides,
  };
}

/// The older segment line (made-up numbers): every present in the mean, nothing left out, no beginning or end said.
export function oldSegmentLine(k, first, last, mean, inForce = 'VICTORY_TEXTURE_LOW=70') {
  return (
    `live segment ${k}: frames ${first}..${last}, mean ${mean.toFixed(1)}, p50 ${mean.toFixed(1)}, ` +
    `p95 ${(mean + 2).toFixed(1)}, skipped 0 draws and 0 dispatches, in force: ${inForce}`
  );
}
