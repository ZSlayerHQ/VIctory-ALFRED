import assert from 'node:assert/strict';
import test from 'node:test';
import {
  collect,
  parseAhead,
  parseBufferCopies,
  parseCapabilities,
  parseLeftOut,
  parseReasonPart,
  parseReasons,
  parseSettleMs,
  parseTableConstants,
  parseThreadClock,
  splitTopLevel,
} from './run-lines.mjs';

// The table-constants line in the runtime's format, with made-up values.
const TABLE_CONSTANTS =
  'table constants per frame (VICTORY_TABLE_CONSTANTS=1): 7 read in place, 2 copied (memory read in place 0, ' +
  'no copy or past it 1, off 16 bytes in its copy 1, in pages rewritten under a reader 0); 1.5 refreshes waited ' +
  '0.4 ms for a submission reading constants in place';

// The settle line's milliseconds by kind in the runtime's format, with made-up values.
const SETTLE_MS =
  'of the waiting for the card in settles, per frame, by what settled: a synchronous submission or present 1.2 ms, ' +
  'the card had run them 0.4, a wait for a memory value 0.0, title memory changed under shader writes 2.5, ' +
  'title memory read 6.1, a step of ours 0.3, lists in the air 0.0, bottom-of-pipe writes 0.7, ' +
  'a blocking fence wait 0.9';

// One reason of the late line with its own breakdown, in the runtime's format, with made-up values.
const READ_REASON = 'title memory read 12.0 (by a buffer copy 7.0, a fill of upload memory 4.0, a texture 1.0)';

test('the capability line parses every field, -1 included, and keeps the text as printed', () => {
  // From the specified format (not yet seen in a log).
  const line =
    'host capabilities: gpu_upload_heaps=1 enhanced_barriers=0 tiled_tier=1 binding_tier=1 ' +
    'dedicated_video_bytes=8152686592 local_budget_bytes=-1';
  const parsed = parseCapabilities(line);
  assert.equal(parsed.fields.gpu_upload_heaps, 1);
  assert.equal(parsed.fields.enhanced_barriers, 0);
  assert.equal(parsed.fields.dedicated_video_bytes, 8152686592);
  assert.equal(parsed.fields.local_budget_bytes, -1);
  assert.match(parsed.raw, /^gpu_upload_heaps=1 /);
  assert.equal(parseCapabilities('host pipeline refused error=x'), null);
});

test('the thread clock carries named threads, other and total', () => {
  // From the specified format.
  const line =
    'card clock threads: frame=120.0 | "victory-copy" 120.5 3.2 41.0 | "title-main" 20.1 0.0 6.7 | ' +
    'other 55.4 12.1 22.5 | total 300.0 15.3 105.0';
  const parsed = parseThreadClock(line);
  assert.equal(parsed.frame, 120);
  assert.deepEqual(parsed.threads[0], { name: 'victory-copy', user: 120.5, kernel: 3.2, pct: 41 });
  assert.deepEqual(parsed.threads[1], { name: 'title-main', user: 20.1, kernel: 0, pct: 6.7 });
  assert.deepEqual(parsed.other, { user: 55.4, kernel: 12.1, pct: 22.5 });
  assert.deepEqual(parsed.total, { user: 300, kernel: 15.3, pct: 105 });
});

test('table constants: read in place, copied, and why the copies happened', () => {
  const parsed = parseTableConstants(TABLE_CONSTANTS);
  assert.equal(parsed.inPlace, 7);
  assert.equal(parsed.copied, 2);
  assert.equal(parsed.reasons['no copy or past it'], 1);
  assert.equal(parsed.waited, 1.5);
  assert.equal(parsed.waitedMs, 0.4);
  // A build without the switch name in the line is still read.
  const plain = parseTableConstants('table constants per frame: 2 read in place, 3 copied (off 16 bytes in its copy 3)');
  assert.equal(plain.copied, 3);
  assert.equal(plain.reasons['off 16 bytes in its copy'], 3);
});

test('buffer copies: the card, the CPU with its reasons, and the flights they settled', () => {
  // From the specified format (not yet seen in a log).
  const line =
    'buffer copies per frame: 12 on the card from host copies (2.1 MB), 6 read from title memory on the CPU ' +
    '(1.0 MB: the switch off 0, off 4 bytes 5, no host copy holding an end 1, memory the card reads in place 0, ' +
    'overlapping ends 0, no copy pipeline 0), 4 of them settled the flights in the air (0.8 MB)';
  const parsed = parseBufferCopies(line);
  assert.equal(parsed.onCard, 12);
  assert.equal(parsed.onCardMb, 2.1);
  assert.equal(parsed.onCpu, 6);
  assert.equal(parsed.onCpuMb, 1);
  assert.equal(parsed.reasons['off 4 bytes'], 5);
  assert.equal(parsed.reasons['no host copy holding an end'], 1);
  assert.equal(parsed.settled, 4);
  assert.equal(parsed.settledMb, 0.8);
});

test('the settle line gives milliseconds per reason, first entry with its unit', () => {
  const parsed = parseSettleMs(SETTLE_MS);
  assert.equal(parsed['a synchronous submission or present'], 1.2);
  assert.equal(parsed['title memory read'], 6.1);
  assert.equal(parsed['a blocking fence wait'], 0.9);
  assert.equal(Object.keys(parsed).length, 9);
});

test('a reason of the late line keeps its count and its own breakdown apart', () => {
  const parsed = parseReasonPart(READ_REASON);
  assert.equal(parsed.reason, 'title memory read');
  assert.equal(parsed.count, 12);
  assert.deepEqual(parsed.by, { 'a buffer copy': 7, 'a fill of upload memory': 4, 'a texture': 1 });
  assert.deepEqual(parseReasonPart('a step of ours 2.0'), { reason: 'a step of ours', count: 2, by: {} });
});

test('a pipeline left out is counted per frame under its reason', () => {
  const line =
    'dropped 12.5/frame without a host pipeline: feeds or traces ray-tracing structures, which are not built; left out';
  assert.deepEqual(parseLeftOut(line), {
    perFrame: 12.5,
    reason: 'feeds or traces ray-tracing structures, which are not built; left out',
  });
});

test('the reason lists split on their own commas only', () => {
  assert.deepEqual(splitTopLevel('a 1, b (x, y) 2, c 3'), ['a 1', 'b (x, y) 2', 'c 3']);
  assert.deepEqual(parseReasons('one 1, two 2'), { one: 1, two: 2 });
});

test('collect reads capabilities anywhere and the per-frame lines only after the start', () => {
  const lines = [
    'host capabilities: gpu_upload_heaps=1 enhanced_barriers=1 tiled_tier=1 binding_tier=1 dedicated_video_bytes=1 local_budget_bytes=1',
    '3.0 fps, 330.0 ms/frame, 5000 draws/frame',
    TABLE_CONSTANTS,
    SETTLE_MS,
    'harness: settled at 75 s (waiting 40, 2.4 fps)',
    TABLE_CONSTANTS,
    'buffer copies per frame: 1 on the card from host copies (0.1 MB), 0 read from title memory on the CPU (0.0 MB: the switch off 0), 0 of them settled the flights in the air (0.0 MB)',
    'dropped 0.5/frame without a host pipeline: too dear to compile, left out',
    'card clock threads: frame=2.0 | "w" 1.0 0.0 0.3 | other 0.0 0.0 0.0 | total 1.0 0.0 0.3',
  ];
  const parsed = collect(lines, 5);
  assert.equal(parsed.capabilities.fields.enhanced_barriers, 1);
  assert.equal(parsed.tableConstants.length, 1);
  assert.equal(parsed.tableConstants[0].inPlace, 7);
  assert.equal(parsed.bufferCopies.length, 1);
  assert.equal(parsed.bufferCopies[0].onCard, 1);
  assert.equal(parsed.settleMs.length, 0);
  assert.deepEqual(parsed.leftOut, [{ perFrame: 0.5, reason: 'too dear to compile, left out' }]);
  assert.equal(parsed.threads[0].threads[0].name, 'w');
});

// With the relay landing flights the line carries no flights-each figure: absent is null, never NaN or zero.
test('the sent-ahead line without a flights-each figure parses it as absent', () => {
  const parsed = parseAhead(
    'submissions sent ahead per frame: 28.5 in 3.2 settles (flights landed by the relay), 41.7 ms host time waiting for settles; settled for: a fill 2.1',
  );
  assert.equal(parsed.flightsEach, null);
  assert.equal(parsed.waiting, 41.7);
});

// The runtime's lines as its own tests pin them (fixtures/runtime-lines.mjs), and its fence watcher's and stats file's
// lines (made-up numbers, in the runtime's text): none of them is a line of this module's kinds, so each parser leaves
// them alone; the pipelines-left-out line keeps its text (only its window grew).
test("the runtime's frame, budget, GPU and segment lines are no line of this module's, and the left-out line still parses", async () => {
  const lines = await import('./run-lines.mjs');
  const shapes = await import('./fixtures/runtime-lines.mjs');
  const P = '2026-11-26T05:02:11.250000Z  INFO victory_gpu::host: ';
  const fresh = [
    `${P}${shapes.FRAME_LINE}`,
    `${P}${shapes.BUDGET_LINE}`,
    `${P}${shapes.BUDGET_LINE_OTHERS}`,
    `${P}${shapes.GPU_LINE}`,
    `${P}our queue per frame, as a watcher of its fence sees it: nothing of ours queued 12.0 ms on the drawing queue (a host-side estimate, not the card's load and not idle time: the copy queue's fetches are not watched), 4.0 ms of that while the thread holding the submission lock was working, the rest while it was not (between submissions or in a host wait); submission-side host waits and fetch latency 3.0 ms`,
    `${P}${shapes.SEGMENT_OPEN_LINE}`,
    `${P}${shapes.SEGMENT_CLOSED_LINE}`,
    `${P}${shapes.MARK_LINE}`,
    `${P}live stats: one JSON object about a second is written here, each covering every frame since the one before it`,
  ];
  const parsers = ['parseCapabilities', 'parseThreadClock', 'parseTableConstants', 'parseBufferCopies', 'parseSettleMs', 'parseLeftOut', 'parseAhead', 'parseNullMagenta', 'parseFirstTouch'];
  for (const line of fresh) {
    for (const name of parsers) assert.equal(lines[name](line), null, `${name} on: ${line.slice(P.length, P.length + 40)}`);
    assert.equal(lines.parsePipelineWait(line), null);
  }
  const collected = lines.collect(fresh);
  for (const key of Object.keys(collected)) {
    if (Array.isArray(collected[key])) assert.equal(collected[key].length, 0, key);
  }
  assert.deepEqual(lines.parseLeftOut(`${P}dropped 10.0/frame without a host pipeline: feeds or traces ray-tracing structures`), {
    perFrame: 10,
    reason: 'feeds or traces ray-tracing structures',
  });
});
