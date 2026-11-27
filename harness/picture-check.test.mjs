import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

import { isolatedEnv } from './test-isolation.mjs';
import { formatDriveReport, reportRows } from './live-drive.mjs';
import {
  analyseImage,
  analyseVideo,
  analyseVideoWindows,
  annotatePictureSegments,
  COULD_NOT_CHECK,
  cropRect,
  DEFAULTS,
  exitCodeFor,
  filterGraph,
  FINDINGS,
  frameClock,
  frameCrop,
  loadCrops,
  loadReference,
  normalizeCrops,
  parseMetadataPrint,
  segmentWindow,
  writeReference,
} from './picture-check.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'picture-check.mjs');

// Every input below is synthetic: grey fields, the validator's gradient (gen.sh), boxes, specks and lavfi sources,
// written as PNGs or encoded with libx264 into a temporary folder. No frame of the game is used or kept.

function tempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'picture-check-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

function frameOf(width, height, data) {
  return { width, height, data };
}

// A grey field, and the same field painted by a callback. `paint` gets `(x, y)` and returns a colour or null.
function grey(width, height, value = 128) {
  return frameOf(width, height, Buffer.alloc(width * height * 3, value));
}

function painted(base, paint) {
  const data = Buffer.from(base.data);
  for (let y = 0; y < base.height; y += 1) {
    for (let x = 0; x < base.width; x += 1) {
      const colour = paint(x, y);
      if (!colour) continue;
      const at = (y * base.width + x) * 3;
      data[at] = colour[0];
      data[at + 1] = colour[1];
      data[at + 2] = colour[2];
    }
  }
  return frameOf(base.width, base.height, data);
}

const black = () => [0, 0, 0];
const box = (left, top, width, height, colour) => (x, y) =>
  x >= left && x < left + width && y >= top && y < top + height ? colour : null;

// The validator's gradient at any size: red 40 to 200 across, green 40 to 220 down, blue 128.
function gradient(width, height) {
  const data = Buffer.alloc(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const at = (y * width + x) * 3;
      data[at] = 40 + Math.floor((x * 160) / width);
      data[at + 1] = 40 + Math.floor((y * 180) / height);
      data[at + 2] = 128;
    }
  }
  return frameOf(width, height, data);
}

// Forty 2x2 black dots: separate small areas rather than one object.
function dotted(base) {
  return painted(base, (x, y) => {
    if (x < 2 || y < 2) return null;
    const col = x - 2;
    const row = y - 2;
    if (col % 9 >= 2 || row % 11 >= 2 || row >= 44 || col >= 90) return null;
    return black();
  });
}

// A PNG ffmpeg decodes: 8-bit RGB, no interlacing, filter 0.
function png(frame) {
  const chunk = (kind, data) => {
    const out = Buffer.alloc(12 + data.length);
    out.writeUInt32BE(data.length, 0);
    out.write(kind, 4, 'latin1');
    data.copy(out, 8);
    const crc = zlib.crc32 ? zlib.crc32(Buffer.concat([Buffer.from(kind, 'latin1'), data])) : 0;
    out.writeUInt32BE(crc >>> 0, 8 + data.length);
    return out;
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(frame.width, 0);
  ihdr.writeUInt32BE(frame.height, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const rows = [];
  for (let y = 0; y < frame.height; y += 1) {
    rows.push(Buffer.from([0]));
    rows.push(frame.data.subarray(y * frame.width * 3, (y + 1) * frame.width * 3));
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(Buffer.concat(rows))),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

function pngFile(dir, name, frame) {
  const file = path.join(dir, name);
  fs.writeFileSync(file, png(frame));
  return file;
}

function ffmpegMake(args) {
  const made = spawnSync('ffmpeg', ['-hide_banner', '-v', 'error', '-y', ...args], { encoding: 'utf8' });
  assert.equal(made.status, 0, made.stderr);
}

const escapeRe = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// One decoded frame of a clip as 8-bit grey, and the mean of a rectangle of it: proof that a generated mover moves.
function greyFrameOf(file, index) {
  const made = spawnSync(
    'ffmpeg',
    ['-hide_banner', '-v', 'error', '-i', file, '-vf', `trim=start_frame=${index}`, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
    { maxBuffer: 64 * 1024 * 1024 },
  );
  assert.equal(made.status, 0, String(made.stderr));
  return made.stdout;
}

function regionMean(frame, width, left, top, w, h) {
  let sum = 0;
  for (let y = top; y < top + h; y += 1) for (let x = left; x < left + w; x += 1) sum += frame[y * width + x];
  return sum / (w * h);
}

function leavesOf(value, out = []) {
  if (Array.isArray(value)) {
    for (const item of value) leavesOf(item, out);
  } else if (value && typeof value === 'object') {
    for (const item of Object.values(value)) leavesOf(item, out);
  } else {
    out.push(value);
  }
  return out;
}

function check(dir, input, extra = []) {
  const result = spawnSync(
    process.execPath,
    [script, 'check', input, '--crops', path.join(dir, 'no-crops.json'), ...extra, '--json'],
    { encoding: 'utf8', env: isolatedEnv() },
  );
  let json = null;
  try {
    json = JSON.parse(result.stdout);
  } catch {
    // an error result prints no JSON
  }
  return { status: result.status, json, stderr: result.stderr };
}

const crop = (result, name = 'frame') => result.frames[0].crops[name];

// ---------------------------------------------------------------------------------------------------------------
// The filter graph and ffmpeg's output

test('the graph runs ffmpeg filters with every option written out, and the metadata print is parsed', () => {
  const graph = filterGraph({
    width: 640,
    height: 360,
    crops: [...frameCrop(), { name: 'flag', x: 380 / 640, y: 40 / 360, w: 80 / 640, h: 80 / 360 }],
    reference: { file: 'reference.png' },
    sample: 1,
    opts: DEFAULTS,
  });
  assert.match(graph, /blackframe=amount=0:threshold=24/);
  assert.match(graph, /freezedetect=n=0\.001:d=5/);
  assert.match(graph, /\[c1\]\[rca1\]ssim\[s1\]/);
  assert.match(graph, /crop=80:80:380:40/);
  assert.match(graph, /metadata=mode=print:file=crop-1\.txt/);

  const parsed = parseMetadataPrint(
    'frame:0    pts:0       pts_time:0\nlavfi.ssim.All=0.5\nlavfi.blackframe.pblack=25\n' +
      'frame:1    pts:1       pts_time:1\nlavfi.psnr.psnr_avg=inf\n',
  );
  assert.deepEqual(parsed, [
    { frame: 0, ptsTime: 0, meta: { 'lavfi.ssim.All': '0.5', 'lavfi.blackframe.pblack': '25' } },
    { frame: 1, ptsTime: 1, meta: { 'lavfi.psnr.psnr_avg': 'inf' } },
  ]);
});

test('a grey field is clean against a reference of itself, and a missing crops file is the whole frame', async (t) => {
  const dir = tempDir(t);
  const frame = pngFile(dir, 'grey.png', grey(100, 50));
  const missing = path.join(dir, 'no-crops.json');
  assert.deepEqual(loadCrops(missing), frameCrop());
  assert.equal(normalizeCrops({ shirt: { x: 0, y: 0, w: 1, h: 1 } })[0].name, 'shirt');
  assert.equal(normalizeCrops({ crops: { chair: { x: 0, y: 0, w: 0.5, h: 0.5 } } })[0].name, 'chair');

  const result = await analyseImage(frame, { crops: loadCrops(missing), reference: { file: frame } });
  assert.equal(result.verdict, 'clean');
  assert.deepEqual(Object.keys(result.frames[0].crops), ['frame']);
  assert.equal(crop(result).ssim, 1);
  assert.equal(crop(result).psnr, null, 'identical: ffmpeg prints an infinite PSNR');
  assert.equal(crop(result).mse, 0);
  assert.equal(crop(result).blackShare, 0);
  assert.equal(result.frozen.verdict, 'clean');
});

// Replaces the home-made dot counter's tests (forty 2x2 dots; forty one-texel specks): SSIM against a fresh
// reference sees them, and the same dotted frame as its own reference is clean.
// Corrected (was `differs` for both against a reference of no stated age, and dots "not a black crop"): differs
// needs a reference within referenceMaxAgeSeconds, here stated as 0 s; the dots, 3.2% of the crop, raise its dark
// share 3 points over the reference's and are black; the specks, 0.8%, stay under a whole point and differ.
test('scattered dots and one-texel specks differ from a clean reference by SSIM, and match themselves', async (t) => {
  const dir = tempDir(t);
  const base = pngFile(dir, 'base.png', grey(100, 50));
  const dots = pngFile(dir, 'dots.png', dotted(grey(100, 50)));
  const specks = pngFile(
    dir,
    'specks.png',
    painted(grey(100, 50), (x, y) => (x % 10 === 5 && y % 10 === 5 && y < 40 ? black() : null)),
  );

  for (const [file, verdict, blackShare] of [[dots, 'black', 0.03], [specks, 'differs', 0]]) {
    const result = await analyseImage(file, { reference: { file: base }, referenceAgeSeconds: 0 });
    assert.equal(result.verdict, verdict, path.basename(file));
    assert.ok(crop(result).ssim < DEFAULTS.ssimMin, `${path.basename(file)}: ssim ${crop(result).ssim}`);
    assert.match(crop(result).reasons.at(-1), /^ssim .* against a reference 0 s away, under 0\.9$/);
    assert.equal(crop(result).blackShare, blackShare);
    assert.equal((await analyseImage(file, { reference: { file }, referenceAgeSeconds: 0 })).verdict, 'clean');
  }
});

// Replaces the home-made black-area and dark tests: a black rectangle and a dim field differ from the reference;
// a crop that is all black is `black` (blackframe), and a dim field is not black.
test('a black rectangle and a dim field differ; a crop inside the rectangle is black by blackframe', async (t) => {
  const dir = tempDir(t);
  // A crop is at least 16x16 (SSIM's window on the half-size chroma planes), so the fields are 200x100.
  const base = pngFile(dir, 'base.png', grey(200, 100));
  const hole = pngFile(dir, 'hole.png', painted(grey(200, 100), box(20, 20, 40, 20, black())));
  const dim = pngFile(dir, 'dim.png', grey(200, 100, 48));
  const crops = [...frameCrop(), { name: 'inside', x: 0.12, y: 0.24, w: 0.16, h: 0.12 }];

  const found = await analyseImage(hole, { crops, reference: { file: base } });
  assert.equal(crop(found, 'inside').verdict, 'black');
  assert.equal(crop(found, 'inside').blackShare, 1);
  // A hole of 4% of the frame moves the whole frame's SSIM only a little (the named limit: a crop finds it).
  assert.ok(crop(found, 'frame').ssim < 1, `ssim ${crop(found, 'frame').ssim}`);
  assert.ok(Math.abs(crop(found, 'frame').blackShare - 0.04) <= 0.01, 'blackframe: 800 of 20000 texels');
  assert.equal(found.verdict, 'black');

  // Corrected (was `differs` against a reference of no stated age): differs needs a fresh reference, stated 0 s.
  const dimmed = await analyseImage(dim, { reference: { file: base }, referenceAgeSeconds: 0 });
  assert.equal(dimmed.verdict, 'differs');
  assert.equal(crop(dimmed).blackShare, 0, 'level 48 is far over the black threshold');
  assert.equal((await analyseImage(dim, { reference: { file: dim } })).verdict, 'clean');
});

// Replaces the share-of-black-under-resize test: blackframe's share of a crop does not move with the frame's size,
// and a reference is scaled to the frames' size, so the same picture at another size is still clean.
test('crop rectangles are fractions: a resized frame keeps its black share and still matches its reference', async (t) => {
  const dir = tempDir(t);
  const paint = (scale) =>
    painted(grey(100 * scale, 50 * scale), box(12 * scale, 12 * scale, 10 * scale, 6 * scale, black()));
  const small = pngFile(dir, 'small.png', paint(2));
  const large = pngFile(dir, 'large.png', paint(4));
  const crops = [{ name: 'seam', x: 0.1, y: 0.2, w: 0.2, h: 0.2 }];
  assert.deepEqual(cropRect(crops[0], 100, 50), { x: 10, y: 10, w: 20, h: 10 });

  const a = await analyseImage(small, { crops, reference: { file: small } });
  const b = await analyseImage(large, { crops, reference: { file: large } });
  assert.equal(crop(a, 'seam').blackShare, 0.3);
  assert.equal(crop(b, 'seam').blackShare, 0.3);
  // Against the smaller reference the frame is measured at the reference's size, never the reference scaled up.
  const c = await analyseImage(large, { crops, reference: { file: small } });
  assert.deepEqual(c.frames[0].measuredAt, { width: 200, height: 100 });
  assert.equal(c.verdict, 'clean', JSON.stringify(crop(c, 'seam')));
});

// Corrected twice (audit 2026-11-26): this test once asserted `clean` for a crop the reference lacked. A reference
// is now a frame, so every crop is cut from it; with no reference at all a crop is `unchecked: no reference`,
// never clean, and an all-black frame is still `black`, since blackframe needs no reference.
test('with no reference a frame is unchecked: no reference, and an all-black frame is black', async (t) => {
  const dir = tempDir(t);
  const plain = await analyseImage(pngFile(dir, 'grey.png', grey(100, 50)));
  assert.equal(plain.verdict, 'unchecked');
  assert.equal(crop(plain).verdict, 'unchecked');
  assert.deepEqual(crop(plain).reasons, ['no reference']);
  assert.equal(crop(plain).ssim, null);
  assert.match(plain.summary, /^picture: unchecked: no reference/);
  assert.equal(plain.reference, false);

  const dark = await analyseImage(pngFile(dir, 'black.png', grey(100, 50, 0)));
  assert.equal(dark.verdict, 'black');
  assert.equal(crop(dark).blackShare, 1);
  assert.match(dark.summary, /^picture: black: no reference/);
});

test('the command JSON is numbers and verdicts only: no pixel array, no path', (t) => {
  const dir = tempDir(t);
  const frameFile = pngFile(dir, 'frame.png', painted(grey(64, 32), box(8, 8, 16, 8, black())));
  const referenceFile = pngFile(dir, 'reference.png', grey(64, 32));
  const cropsFile = path.join(dir, 'crops.json');
  fs.writeFileSync(cropsFile, JSON.stringify({ crops: [{ name: 'centre', x: 0.1, y: 0.2, w: 0.5, h: 0.5 }] }));

  const result = spawnSync(
    process.execPath,
    [script, 'check', frameFile, '--crops', cropsFile, '--reference', referenceFile, '--reference-age', '0', '--json'],
    { encoding: 'utf8', env: isolatedEnv() },
  );
  // Corrected (audit 2026-11-26): this once asserted exit 0 for a finding; findings exit 1.
  assert.equal(result.status, 1, result.stderr);
  const parsed = JSON.parse(result.stdout);
  // Corrected (was `differs`, against a reference of no stated age, now stated 0 s): the box is a quarter of the
  // crop, 25 points of dark share over the fresh reference's: black.
  assert.equal(parsed.verdict, 'black');
  assert.equal(parsed.frames[0].crops.centre.verdict, 'black');
  assert.equal(parsed.tool, 'ffmpeg');
  const json = JSON.stringify(parsed);
  assert.doesNotMatch(json, /"data"\s*:/);
  assert.doesNotMatch(json, /"pixels?"\s*:/);
  assert.doesNotMatch(json, new RegExp(escapeRe(dir)));
  assert.doesNotMatch(json, /[A-Za-z]:[\\/]/);
  assert.ok(
    leavesOf(parsed).every((value) => value === null || ['number', 'string', 'boolean'].includes(typeof value)),
    'only scalar leaves, never a Buffer or an array of samples',
  );
});

// Replaces the numbers-only reference test: the reference is now a frame written by ffmpeg.
test('the reference command writes a frame from an image or a recording; check compares against it', (t) => {
  const dir = tempDir(t);
  const frameFile = pngFile(dir, 'good.png', painted(grey(64, 32), box(12, 10, 8, 4, black())));
  const out = path.join(dir, 'ref', 'reference.png');
  const made = spawnSync(process.execPath, [script, 'reference', '--frame', frameFile, '--out', out], {
    encoding: 'utf8',
    env: isolatedEnv(),
  });
  assert.equal(made.status, 0, made.stderr);
  assert.match(made.stdout, /^picture reference: a 64x32 frame written/);
  assert.doesNotMatch(made.stdout, new RegExp(escapeRe(dir)));
  assert.ok(fs.readFileSync(out).subarray(1, 4).equals(Buffer.from('PNG')), 'a PNG frame, beside nothing else');
  assert.deepEqual(loadReference(out), { file: path.resolve(out) });

  const checked = check(dir, frameFile, ['--reference', out]);
  assert.equal(checked.status, 0, checked.stderr);
  assert.equal(checked.json.verdict, 'clean');

  const video = path.join(dir, 'rec.mp4');
  ffmpegMake(['-f', 'lavfi', '-i', 'color=c=gray:s=64x32:r=2:d=3', '-pix_fmt', 'yuv420p', video]);
  const fromVideo = spawnSync(
    process.execPath,
    [script, 'reference', '--video', video, '--at', '1', '--out', path.join(dir, 'from-video.png')],
    { encoding: 'utf8', env: isolatedEnv() },
  );
  assert.equal(fromVideo.status, 0, fromVideo.stderr);
  const noAt = spawnSync(process.execPath, [script, 'reference', '--video', video], { encoding: 'utf8', env: isolatedEnv() });
  assert.equal(noAt.status, 2);
});

test('a live segment frame range becomes recording seconds through the log frame lines', () => {
  const lines = [
    'recording: started 2026-11-26T00:00:00.000Z',
    '2026-11-26T00:00:01.000000Z  INFO victory_gpu::host: frame 100, 20.0 fps, 50.0 ms/frame (0 preparing: copies 0.0 ms 0.0 MB, textures 0.0 ms 0.0 MB; 0.0 on the GPU; 0.0 copying back 0.0 MB), 10 draws/frame, dropped 0 without a host pipeline (0 being made) and 0 indirect',
    '2026-11-26T00:00:06.000000Z  INFO victory_gpu::host: frame 200, 20.0 fps, 50.0 ms/frame (0 preparing: copies 0.0 ms 0.0 MB, textures 0.0 ms 0.0 MB; 0.0 on the GPU; 0.0 copying back 0.0 MB), 10 draws/frame, dropped 0 without a host pipeline (0 being made) and 0 indirect',
  ];
  const clock = frameClock(lines);
  assert.equal(clock.startedAt, Date.parse('2026-11-26T00:00:00.000Z'));
  assert.deepEqual(
    clock.samples.map((sample) => sample.frame),
    [100, 200],
  );
  const window = segmentWindow({ firstFrame: 150, lastFrame: 190 }, clock);
  assert.ok(Math.abs(window.fromSeconds - 3.5) < 0.001);
  assert.ok(Math.abs(window.toSeconds - 5.55) < 0.01);
});
// A four-second recording: two seconds of grey, then two of black, 64x32 at two frames a second.
function recording(dir) {
  const capture = path.join(dir, 'capture');
  fs.mkdirSync(capture, { recursive: true });
  const made = spawnSync(
    'ffmpeg',
    [
      '-hide_banner',
      '-v',
      'error',
      '-y',
      '-f',
      'lavfi',
      '-i',
      'color=c=gray:s=64x32:r=2:d=2',
      '-f',
      'lavfi',
      '-i',
      'color=c=black:s=64x32:r=2:d=2',
      '-filter_complex',
      '[0:v][1:v]concat=n=2:v=1:a=0',
      '-pix_fmt',
      'yuv420p',
      path.join(capture, 'screen.mp4'),
    ],
    { encoding: 'utf8' },
  );
  assert.equal(made.status, 0, made.stderr);
  return capture;
}

// The captures root with a reference frame: a grey frame of the recording's size.
function referenceRoot(dir) {
  const root = path.join(dir, 'caps');
  fs.mkdirSync(path.join(root, 'reference'), { recursive: true });
  fs.writeFileSync(path.join(root, 'reference', 'reference.png'), png(grey(64, 32)));
  return root;
}

function summaryLog(dir, capture) {
  const frame = (ts, number) =>
    `${ts}  INFO victory_gpu::host: frame ${number}, 2.0 fps, 500.0 ms/frame (0 preparing: copies 0.0 ms 0.0 MB, ` +
    'textures 0.0 ms 0.0 MB; 0.0 on the GPU; 0.0 copying back 0.0 MB), 10 draws/frame, dropped 0 without a host ' +
    'pipeline (0 being made) and 0 indirect';
  const log = path.join(dir, 'run.log');
  fs.writeFileSync(
    log,
    [
      `capture: ${capture}`,
      'recording: started 2026-11-26T00:00:00.000Z',
      frame('2026-11-26T00:00:00.000000Z', 0),
      frame('2026-11-26T00:00:04.000000Z', 40),
      'harness: victory exit=0',
      '',
    ].join('\n'),
  );
  return log;
}

// summarize-run still names `reference.json`; the reference frame beside it is found. Its per-crop line prints
// fields of the old detector, which summarize-run's owner updates; this test pins the verdicts only.
test('summarize-run gains a picture block from a recording and a reference frame', (t) => {
  const dir = tempDir(t);
  const capture = recording(dir);
  const root = referenceRoot(dir);
  const log = summaryLog(dir, capture);
  const env = isolatedEnv({ VICTORY_CAPTURES: root });

  const text = spawnSync(process.execPath, [path.join(here, 'summarize-run.mjs'), log], { encoding: 'utf8', env });
  assert.equal(text.status, 0, text.stderr);
  assert.match(text.stdout, /## Picture/);
  assert.match(text.stdout, /- verdict: black/);
  assert.match(text.stdout, /worst frame/);

  const json = spawnSync(process.execPath, [path.join(here, 'summarize-run.mjs'), log, '--json'], {
    encoding: 'utf8',
    env,
  });
  assert.equal(json.status, 0, json.stderr);
  const picture = JSON.parse(json.stdout).picture;
  assert.equal(picture.verdict, 'black');
  assert.equal(picture.reference, true);
  assert.equal(picture.crops.frame.verdict, 'black');
  assert.ok(picture.worstFrame);
  assert.doesNotMatch(JSON.stringify(picture), new RegExp(escapeRe(dir)));
});

test('the plan runner reads a step segment from the recording and rejects the one that turns black', async (t) => {
  const dir = tempDir(t);
  const capture = recording(dir);
  const root = referenceRoot(dir);
  const log = summaryLog(dir, capture);
  const segment = (number, firstFrame, lastFrame) => ({
    segment: number,
    firstFrame,
    lastFrame,
    frameCount: lastFrame - firstFrame + 1,
    mean: 20,
    p50: 20,
    p95: 22,
    skippedDraws: 0,
    skippedDispatches: 0,
    inForce: {},
    short: true,
    meanChange: null,
    changes: [],
  });
  const segments = [segment(1, 1, 5), segment(2, 20, 39)];
  const occurrence = (label, stepSegment) => ({
    label,
    set: {},
    pass: 1,
    direction: 'forward',
    beforeSegment: null,
    stepSegment,
    afterSegment: null,
  });
  const occurrences = [occurrence('low', 1), occurrence('skip', 2)];

  const note = await annotatePictureSegments({
    logFile: log,
    segments,
    occurrences,
    root,
    stepSeconds: 1,
    maxWidth: 64,
  });
  assert.equal(note, '2 segments checked');
  assert.equal(occurrences[0].picture.verdict, 'clean');
  assert.equal(occurrences[0].rejected, undefined);
  // Corrected (was 'black-area'): the verdict is blackframe's `black` since the detectors became ffmpeg's filters.
  assert.equal(occurrences[1].picture.verdict, 'black');
  assert.deepEqual(occurrences[1].rejected, ['picture']);

  // Corrected (was a version 1 sidecar, whose rows are now all unreliable and carry nothing): a version 2 sidecar.
  // Its arms are not under test here; the rows are unreliable for want of them and still carry the picture.
  const timeline = { version: 2, touched: [], start: {}, arms: [], occurrences };
  const rows = reportRows(segments, timeline);
  assert.equal(rows.find((row) => row.label === 'low').picture.verdict, 'clean');
  assert.equal(rows.find((row) => row.label === 'skip').picture.verdict, 'black');
  assert.deepEqual(rows.find((row) => row.label === 'skip').rejected, ['picture']);
  const report = formatDriveReport(rows, timeline);
  assert.match(report, /\n- skip: black — rejected: picture/);
});

// ---------------------------------------------------------------------------------------------------------------
// What the tool says when it could not check (audit 2026-11-26 and the validator's probes, rebuilt here from
// synthetic inputs: a grey field, a recording of it, a fragmented one cut inside a fragment, junk bytes).

// Ten seconds of grey, 64x32 at two frames a second, written as the harness writes its recording: fragmented, a
// keyframe every four frames. `fragmented: false` writes an ordinary mp4 instead.
function greyRecording(dir, name, { fragmented = true } = {}) {
  const file = path.join(dir, name);
  ffmpegMake([
    '-f', 'lavfi', '-i', 'color=c=gray:s=64x32:r=2:d=10',
    '-c:v', 'libx264', '-threads', '1', '-g', '4', '-pix_fmt', 'yuv420p',
    ...(fragmented ? ['-movflags', '+frag_keyframe+empty_moov+default_base_moof'] : []),
    file,
  ]);
  return file;
}

// The fragmented recording cut half-way through the middle fragment's media: what a reader sees while ffmpeg is
// still writing it, or after the writer died.
function cutRecording(dir) {
  const whole = fs.readFileSync(greyRecording(dir, 'whole.mp4'));
  const boxes = [];
  for (let at = 0; at + 8 <= whole.length; ) {
    const size = whole.readUInt32BE(at);
    boxes.push({ at, size, type: whole.toString('latin1', at + 4, at + 8) });
    if (size < 8) break;
    at += size;
  }
  const media = boxes.filter((entry) => entry.type === 'mdat');
  assert.ok(media.length >= 4, 'the recording has several fragments');
  const middle = media[Math.floor(media.length / 2)];
  const file = path.join(dir, 'cut.mp4');
  fs.writeFileSync(file, whole.subarray(0, middle.at + Math.floor(middle.size / 2)));
  return file;
}

function greyReference(dir) {
  return pngFile(dir, 'grey-reference.png', grey(64, 32));
}

// The still grey recordings are frozen by freezedetect's rule; tests about coverage set the freeze out of reach.
const NO_FREEZE = 1000;

test('a window outside the recording is no frames, never clean, and the runner does not count it as checked', async (t) => {
  const dir = tempDir(t);
  const video = greyRecording(dir, 'grey.mp4', { fragmented: false });
  const reference = { file: greyReference(dir) };

  const [inside, outside] = await analyseVideoWindows(
    video,
    [
      { fromSeconds: 1, toSeconds: 3 },
      { fromSeconds: 50, toSeconds: 60 },
    ],
    { crops: frameCrop(), reference, stepSeconds: 1, maxWidth: 64, freezeSeconds: NO_FREEZE },
  );
  assert.equal(inside.verdict, 'clean');
  assert.equal(outside.frames.length, 0);
  assert.equal(outside.verdict, 'no frames');
  assert.equal(outside.summary, 'picture: no frames');
  assert.equal(outside.worstFrame, null);

  // The command line: --from past the end of a ten-second recording.
  const late = check(dir, video, ['--reference', reference.file, '--from', '100']);
  assert.equal(late.json?.verdict, 'no frames', late.stderr);
  assert.equal(late.status, 3);

  // The plan runner's path: a segment placed after the recording ends gets `no frames` and is not counted. The
  // early segment is checked: ten seconds of a still grey field is a freeze by freezedetect's rule.
  const capture = path.join(dir, 'capture');
  fs.mkdirSync(capture);
  fs.copyFileSync(video, path.join(capture, 'screen.mp4'));
  const root = referenceRoot(dir);
  const frameLine = (ts, number) =>
    `${ts}  INFO victory_gpu::host: frame ${number}, 10.0 fps, 100.0 ms/frame (0 preparing: copies 0.0 ms 0.0 MB, ` +
    'textures 0.0 ms 0.0 MB; 0.0 on the GPU; 0.0 copying back 0.0 MB), 10 draws/frame, dropped 0 without a host ' +
    'pipeline (0 being made) and 0 indirect';
  const log = path.join(dir, 'run.log');
  fs.writeFileSync(
    log,
    [
      `capture: ${capture}`,
      'recording: started 2026-11-26T00:00:00.000Z',
      frameLine('2026-11-26T00:00:00.000000Z', 0),
      frameLine('2026-11-26T00:01:00.000000Z', 600),
      '',
    ].join('\n'),
  );
  const segments = [
    { segment: 1, firstFrame: 10, lastFrame: 80 },
    { segment: 2, firstFrame: 400, lastFrame: 500 },
  ];
  const occurrences = [
    { label: 'early', set: {}, stepSegment: 1 },
    { label: 'late', set: {}, stepSegment: 2 },
  ];
  const note = await annotatePictureSegments({ logFile: log, segments, occurrences, root, stepSeconds: 1, maxWidth: 64 });
  assert.equal(occurrences[0].picture.verdict, 'frozen');
  assert.equal(occurrences[1].picture.verdict, 'no frames');
  assert.equal(occurrences[1].rejected, undefined, 'no frames is not a picture fault');
  assert.equal(note, '1 segment checked; not checked: 1 no frames');
});

test('a recording cut inside a fragment or still growing is incomplete, not clean; a window it covers still checks', async (t) => {
  const dir = tempDir(t);
  const reference = greyReference(dir);
  const cut = cutRecording(dir);
  const still = ['--freeze-seconds', String(NO_FREEZE)];

  const open = check(dir, cut, ['--reference', reference, ...still]);
  assert.equal(open.json?.verdict, 'incomplete', open.stderr);
  assert.ok(open.json.frames.length > 0, 'the part written was read');
  assert.match(open.json.summary, /^picture: incomplete/);
  assert.equal(open.status, 3);

  const early = check(dir, cut, ['--reference', reference, '--to', '2', ...still]);
  assert.equal(early.json?.verdict, 'clean', early.stderr);
  assert.equal(early.status, 0);

  const options = { crops: frameCrop(), reference: { file: reference }, stepSeconds: 1, maxWidth: 64, freezeSeconds: NO_FREEZE };
  const [covered, past] = await analyseVideoWindows(
    cut,
    [
      { fromSeconds: 0, toSeconds: 2 },
      { fromSeconds: 3, toSeconds: 9.5 },
    ],
    options,
  );
  assert.equal(covered.verdict, 'clean');
  assert.equal(past.verdict, 'incomplete');

  // A whole recording whose size changes while it is read is still being written.
  const plain = greyRecording(dir, 'plain.mp4', { fragmented: false });
  let size = 1000;
  const growing = await analyseVideo(plain, { ...options, statFile: () => ({ size: (size += 500), mtimeMs: size }) });
  assert.equal(growing.verdict, 'incomplete');
  assert.equal((await analyseVideo(plain, options)).verdict, 'clean', 'the same recording, finished, is clean');
});

test('the exit code says clean (0), findings (1), bad arguments (2) or could not check (3)', (t) => {
  const dir = tempDir(t);
  const reference = greyReference(dir);
  const cleanFile = pngFile(dir, 'grey.png', grey(64, 32));
  const holeFile = pngFile(dir, 'hole.png', painted(grey(64, 32), box(8, 8, 32, 16, black())));
  const blackFile = pngFile(dir, 'black.png', grey(64, 32, 0));
  const junkPng = path.join(dir, 'junk.png');
  fs.writeFileSync(junkPng, 'not a picture\n');
  const junkMp4 = path.join(dir, 'junk.mp4');
  fs.writeFileSync(junkMp4, 'not a recording\n');
  const emptyMp4 = path.join(dir, 'empty.mp4');
  fs.writeFileSync(emptyMp4, '');

  assert.equal(check(dir, cleanFile, ['--reference', reference]).status, 0);
  // Corrected (was exit 1 against a reference of no stated age): a hole a quarter of the frame, whose SSIM no
  // longer matches, is a finding against a fresh reference (black, exit 1) and could-not-check against an unaged
  // one (unlike the reference frame, exit 3).
  assert.equal(check(dir, holeFile, ['--reference', reference, '--reference-age', '0']).status, 1);
  assert.equal(check(dir, holeFile, ['--reference', reference]).status, 3);
  assert.equal(check(dir, cleanFile, ['--reference', reference, '--step', 'x']).status, 2);

  // No reference file: a grey frame is unchecked; an all-black one is black, which blackframe needs no reference for.
  const none = check(dir, cleanFile, ['--reference', path.join(dir, 'no-such-reference.png')]);
  assert.equal(none.json?.verdict, 'unchecked');
  assert.equal(none.status, 3);
  const dark = check(dir, blackFile, ['--reference', path.join(dir, 'no-such-reference.png')]);
  assert.equal(dark.json?.verdict, 'black');
  assert.equal(dark.status, 1);

  // A crop is cut from the reference frame too, so a crops file always compares (the validator's flag crop).
  const flagCrops = path.join(dir, 'flag-crops.json');
  fs.writeFileSync(flagCrops, JSON.stringify({ crops: [{ name: 'flag', x: 0.5, y: 0.1, w: 0.25, h: 0.5 }] }));
  const named = spawnSync(
    process.execPath,
    [script, 'check', blackFile, '--crops', flagCrops, '--reference', reference, '--json'],
    { encoding: 'utf8', env: isolatedEnv() },
  );
  assert.equal(JSON.parse(named.stdout).frames[0].crops.flag.verdict, 'black');
  assert.equal(named.status, 1);

  for (const unreadable of [junkPng, junkMp4, emptyMp4, path.join(dir, 'missing.mp4')]) {
    const result = check(dir, unreadable, ['--reference', reference]);
    assert.equal(result.status, 3, `${path.basename(unreadable)}: ${result.stderr}`);
    assert.equal(result.json, null);
    assert.doesNotMatch(result.stderr, new RegExp(escapeRe(dir)));
  }
});

// ---------------------------------------------------------------------------------------------------------------
// The validator's synthetic cases (gen.sh) through the whole path: the gradient (darkest channel 40), boxes on it,
// isolated specks, an all-black frame, a night frame dark everywhere but never black, and motion.

const GRAD = "geq=r='40+X/4':g='40+Y/2':b='128'";
const NIGHT = "geq=r='14+X/40':g='14+Y/30':b='20'";
const FLAG_CROP = [{ name: 'flag', x: 380 / 640, y: 40 / 360, w: 80 / 640, h: 80 / 360 }];

// One 640x360 frame from a filter, through libx264 as the recording is.
function encodedFrame(dir, name, filter) {
  const file = path.join(dir, name);
  ffmpegMake([
    '-f', 'lavfi', '-i', 'color=c=gray:s=640x360:d=1', '-vf', filter, '-frames:v', '1',
    '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', file,
  ]);
  return file;
}

// Replaces the flat-and-darker-than-its-ring test: a darkness threshold cannot see a box the encoder lifts to
// level 20 (blackframe finds none of it black), and SSIM against a reference of the same view can.
test('a box at level 20 through libx264 is not black to blackframe but differs by SSIM; a night frame is not black', async (t) => {
  const dir = tempDir(t);
  const grad = encodedFrame(dir, 'grad.mp4', GRAD);
  const box20 = encodedFrame(dir, 'box20.mp4', `${GRAD},drawbox=x=400:y=60:w=40:h=40:color=0x141414:t=fill`);
  const night = encodedFrame(dir, 'night.mp4', NIGHT);
  const reference = { file: grad };

  // Each is one encoded frame: checked as an image, as the command does for a frame.
  // Corrected (was `differs` and `differs` below against a reference of no stated age): differs needs a fresh
  // reference, stated here as 0 s; the same frames against an unaged one are unlike the reference frame.
  const fresh = { reference, referenceAgeSeconds: 0 };
  const found = await analyseImage(box20, { crops: FLAG_CROP, ...fresh });
  const flag = crop(found, 'flag');
  assert.equal(flag.blackShare, 0, 'the box sits over the black threshold');
  assert.ok(flag.ssim < DEFAULTS.ssimMin, `ssim ${flag.ssim}`);
  assert.equal(found.verdict, 'differs');
  assert.equal((await analyseImage(box20, { crops: FLAG_CROP, reference })).verdict, 'unlike the reference frame');

  // The false-positive trap: dark everywhere and never black.
  const dimNone = await analyseImage(night);
  assert.equal(crop(dimNone).blackShare, 0);
  assert.equal(dimNone.verdict, 'unchecked');
  const dimGrad = await analyseImage(night, fresh);
  assert.equal(dimGrad.verdict, 'differs', 'another picture than the reference');
  assert.equal(crop(dimGrad).referenceBlackShare, 0);
  assert.equal((await analyseImage(night, { reference: { file: night } })).verdict, 'clean');
  assert.equal((await analyseImage(grad, { reference })).verdict, 'clean');

  // Every option passed to a filter is in the output.
  // Corrected (was without mpdecimate, blackRise, referenceMaxAgeSeconds and mpdecimate's three thresholds).
  assert.deepEqual(found.filters, {
    blackframe: 'blackframe=amount=0:threshold=24',
    freezedetect: 'freezedetect=n=0.001:d=5',
    mpdecimate: 'mpdecimate=hi=768:lo=320:frac=0.33',
    ssim: 'ssim',
    psnr: 'psnr',
  });
  assert.deepEqual(found.thresholds, {
    blackThreshold: 24, blackAmount: 98, blackRise: 1, ssimMin: 0.9, referenceMaxAgeSeconds: 1, freezeNoise: 0.001,
    freezeSeconds: 5, decimateHi: 768, decimateLo: 320, decimateFrac: 0.33, stepSeconds: 1,
  });
});

// Replaces the black-area-floor test: the validator's boxes and the all-black frame, whole-frame and in a crop.
// Corrected (was: both boxes `clean` in the whole frame, the 1.3% one included, and the 40x40 box `differs` in its
// crop against a reference of no stated age): a crop's dark share over the reference's is black, so the 1.3% box
// is black in the whole frame (1 point, the rest matching) and the 40x40 box in its crop against a fresh reference
// (25 points); only the 0.7% box, under a whole point, stays clean in the whole frame.
test('the validator boxes: an all-black frame is black; a 40x40 box needs a crop around it to be found', async (t) => {
  const dir = tempDir(t);
  const grad = pngFile(dir, 'grad.png', gradient(640, 360));
  const reference = { file: grad };
  const flag = pngFile(dir, 'flag.png', painted(gradient(640, 360), box(400, 60, 40, 40, black())));
  const rect = pngFile(dir, 'rect.png', painted(gradient(640, 360), box(100, 100, 64, 48, black())));
  const allBlack = pngFile(dir, 'black.png', grey(640, 360, 0));
  const dots = pngFile(dir, 'dots.png', painted(gradient(640, 360), (x, y) => (x % 40 === 20 && y % 40 === 20 ? black() : null)));

  assert.equal((await analyseImage(allBlack, { reference })).verdict, 'black');
  assert.equal((await analyseImage(allBlack, { crops: FLAG_CROP, reference })).verdict, 'black');

  // A box of 0.7% or 1.3% of the frame moves the whole frame's SSIM by about a percent; the dark share sees 1.3%.
  for (const [file, verdict, blackShare] of [[flag, 'clean', 0], [rect, 'black', 0.01]]) {
    const whole = await analyseImage(file, { reference });
    assert.ok(crop(whole).ssim > DEFAULTS.ssimMin && crop(whole).ssim < 1, `${path.basename(file)}: ${crop(whole).ssim}`);
    assert.equal(crop(whole).blackShare, blackShare, path.basename(file));
    assert.equal(whole.verdict, verdict, path.basename(file));
  }
  const inCrop = await analyseImage(flag, { crops: FLAG_CROP, reference, referenceAgeSeconds: 0 });
  assert.equal(crop(inCrop, 'flag').verdict, 'black');
  assert.equal(crop(inCrop, 'flag').blackShare, 0.25, 'blackframe: the box is a quarter of the crop');

  // Sparse one-texel specks (the validator's dots) barely move SSIM on a textured picture: over the default 0.9,
  // under a crop's own ssimMin near 1. The dot counter that saw them is gone; a seam needs a tight crop.
  // Corrected (was `differs` against a reference of no stated age): differs needs a fresh one, stated 0 s.
  const dotCrop = [{ name: 'dots', x: 0, y: 0, w: 0.125, h: 0.2222 }];
  const sparse = await analyseImage(dots, { crops: dotCrop, reference });
  assert.ok(crop(sparse, 'dots').ssim < 1 && crop(sparse, 'dots').ssim > DEFAULTS.ssimMin, `ssim ${crop(sparse, 'dots').ssim}`);
  const tight = await analyseImage(dots, { crops: [{ ...dotCrop[0], ssimMin: 0.99 }], reference, referenceAgeSeconds: 0 });
  assert.equal(tight.verdict, 'differs', `ssim ${crop(tight, 'dots').ssim}`);
  assert.equal((await analyseImage(grad, { crops: [{ ...dotCrop[0], ssimMin: 0.99 }], reference })).verdict, 'clean');
});

// Replaces the relative-margin and measured-at-the-reference-size tests of the dot counter: good frames of one view
// that differ by a few specks stay over the SSIM threshold, and many more specks do not.
test('a few more specks than the reference stay clean by SSIM; many more differ', async (t) => {
  const dir = tempDir(t);
  const specks = (count) =>
    painted(grey(400, 200), (x, y) => (y % 10 === 5 && x % 10 === 5 && (y / 10 - 0.5) * 40 + (x / 10 - 0.5) < count ? black() : null));
  const reference = { file: pngFile(dir, 'ref.png', specks(194)) };
  const near = await analyseImage(pngFile(dir, 'near.png', specks(210)), { reference });
  assert.equal(near.verdict, 'clean', `ssim ${crop(near).ssim}`);
  // Corrected (was `differs` against a reference of no stated age): differs needs a fresh one, stated 0 s.
  const many = await analyseImage(pngFile(dir, 'many.png', specks(700)), { reference, referenceAgeSeconds: 0 });
  assert.equal(many.verdict, 'differs', `ssim ${crop(many).ssim}`);
});

// Replaces the home-made frozen tests: freezedetect and mpdecimate on the whole frame. A still recording is frozen;
// a 16 px box moving over a still frame, and two alternating fields, are not.
test('freezedetect: a still recording is frozen; a small box moving over a still frame is not', async (t) => {
  const dir = tempDir(t);
  const grad = "geq=r='40+X/2':g='40+Y':b='128'";
  const make = (name, moving) => {
    const file = path.join(dir, name);
    ffmpegMake([
      '-f', 'lavfi', '-i', 'color=c=gray:s=320x180:r=2:d=10', '-f', 'lavfi', '-i', 'color=c=white:s=16x16:r=2:d=10',
      '-filter_complex', `[0:v]${grad}[bg];[bg][1:v]overlay=x='${moving ? '10+t*10' : '10'}':y=100:shortest=1`,
      '-c:v', 'libx264', '-threads', '1', '-g', '4', '-pix_fmt', 'yuv420p', file,
    ]);
    return file;
  };
  const stillFile = make('still.mp4', false);
  const motionFile = make('motion.mp4', true);
  // Corrected (was: nothing checked that the mover moves; round 1's validator clip, drawbox with `t`, which is
  // thickness, was byte-identical to its still). This overlay clip does move: where the box starts is white in the
  // first frame (0) and the gradient in the last (19), and the still's box stays put.
  const where = (file, index) => regionMean(greyFrameOf(file, index), 320, 10, 100, 16, 16);
  assert.ok(where(motionFile, 0) - where(motionFile, 19) > 50, 'the box leaves its start');
  assert.ok(Math.abs(where(stillFile, 0) - where(stillFile, 19)) < 5, 'the still box stays');
  const frozen = await analyseVideo(stillFile, { maxWidth: 320 });
  assert.equal(frozen.frozen.verdict, 'frozen');
  assert.equal(frozen.verdict, 'frozen');
  assert.ok(frozen.frozen.seconds >= DEFAULTS.freezeSeconds, `${frozen.frozen.seconds} s`);
  const motion = await analyseVideo(motionFile, { maxWidth: 320 });
  assert.equal(motion.frozen.verdict, 'clean');
  assert.equal(motion.verdict, 'unchecked');

  const flicker = path.join(dir, 'flicker.mp4');
  ffmpegMake([
    '-f', 'lavfi', '-i', "color=c=gray:s=64x32:r=1:d=8,geq=lum='if(mod(N,2),20,200)':cb=128:cr=128",
    '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', flicker,
  ]);
  assert.equal((await analyseVideo(flicker, { maxWidth: 64 })).frozen.verdict, 'clean');
});

// The tool check of 2026-11-26 (check-tools-truth.md, Picture): an 8 px mover read frozen. freezedetect's `n` is a
// mean over the whole frame: an 8 px box crossing 640x360 moves it by 0.0001 to 0.0003, as much as libx264's noise
// on a still textured frame with a keyframe every 2 s, so no `n` separates the two. Frozen now also needs
// mpdecimate, ffmpeg's per-8x8-block duplicate test, to find every block unchanged.
test('an 8 px box crossing a still 640x360 frame is not frozen; a still textured frame through libx264 is', async (t) => {
  const dir = tempDir(t);
  const mover = path.join(dir, 'mover8.mp4');
  ffmpegMake([
    '-f', 'lavfi', '-i', 'color=c=gray:s=640x360:r=10:d=10', '-f', 'lavfi', '-i', 'color=c=white:s=8x8:r=10:d=10',
    '-filter_complex', `[0:v]${GRAD}[bg];[bg][1:v]overlay=x='20+t*20':y=200`,
    '-t', '10', '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', mover,
  ]);
  // The mover moves: where the 8 px box starts is white in the first frame and the gradient in the last.
  const where = (index) => regionMean(greyFrameOf(mover, index), 640, 20, 200, 8, 8);
  assert.ok(where(0) - where(99) > 50, `${where(0)} then ${where(99)}`);
  const moved = await analyseVideo(mover);
  assert.equal(moved.frozen.verdict, 'clean', moved.summary);
  assert.notEqual(moved.verdict, 'frozen');

  const still = path.join(dir, 'still-texture.mp4');
  ffmpegMake([
    '-f', 'lavfi', '-i', 'testsrc2=s=640x360:r=10:d=1', '-vf', 'trim=end_frame=1,loop=-1:1:0,setpts=N/10/TB',
    '-t', '12', '-r', '10', '-c:v', 'libx264', '-threads', '1', '-g', '20', '-pix_fmt', 'yuv420p', still,
  ]);
  const stopped = await analyseVideo(still);
  assert.equal(stopped.frozen.verdict, 'frozen', stopped.summary);
  assert.ok(stopped.frozen.seconds >= DEFAULTS.freezeSeconds, `${stopped.frozen.seconds} s`);
  assert.equal(stopped.filters.mpdecimate, 'mpdecimate=hi=768:lo=320:frac=0.33');
  assert.equal(stopped.filters.freezedetect, 'freezedetect=n=0.001:d=5');
});

// The tool check of 2026-11-26 (Picture): against its own frame of the same view, a real recording read SSIM 0.92
// three seconds away and 0.48 to 0.89 thirty or more away, and 24 of 31 good frames were called `differs`; its
// crops read 0.60 to 0.84 at two seconds. A crop now differs only against a reference taken within
// `referenceMaxAgeSeconds` (1 s) of the frame; against an older or unaged one it keeps its figures and says
// `unlike the reference frame`, which is could-not-check.
test('differs needs a reference within a few seconds of the frame; an older or unaged one is unlike the reference frame', async (t) => {
  const dir = tempDir(t);
  // Five seconds of the gradient, then five of the night picture; the reference is the recording's frame at 5 s.
  const video = path.join(dir, 'two-views.mp4');
  ffmpegMake([
    '-f', 'lavfi', '-i', 'color=c=gray:s=160x90:r=2:d=5', '-f', 'lavfi', '-i', 'color=c=gray:s=160x90:r=2:d=5',
    '-filter_complex', `[0:v]${GRAD}[a];[1:v]${NIGHT}[b];[a][b]concat=n=2:v=1:a=0`,
    '-c:v', 'libx264', '-threads', '1', '-g', '4', '-pix_fmt', 'yuv420p', video,
  ]);
  const ref = path.join(dir, 'ref-at-5.png');
  writeReference({ video, atSeconds: 5, out: ref });
  const options = { reference: { file: ref }, stepSeconds: 1, maxWidth: 160, freezeSeconds: NO_FREEZE };

  const dated = await analyseVideo(video, { ...options, referenceAtSeconds: 5 });
  const at = Object.fromEntries(dated.frames.map((frame) => [frame.timeSeconds, frame.crops.frame]));
  assert.deepEqual(Object.keys(at).map(Number), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  for (const seconds of [0, 1, 2, 3]) {
    const entry = at[seconds];
    assert.equal(entry.verdict, 'unlike the reference frame', `${seconds} s: ${JSON.stringify(entry)}`);
    assert.equal(entry.referenceAgeSeconds, 5 - seconds);
    assert.ok(entry.ssim < DEFAULTS.ssimMin && Number.isFinite(entry.psnr), 'the figures stay');
    assert.match(entry.reasons[0], new RegExp(`a reference ${5 - seconds} s away.*within 1 s`));
  }
  assert.equal(at[4].verdict, 'differs', JSON.stringify(at[4]));
  for (const seconds of [5, 6, 7, 8, 9]) assert.equal(at[seconds].verdict, 'clean', `${seconds} s`);
  assert.equal(dated.verdict, 'differs');
  assert.equal(dated.thresholds.referenceMaxAgeSeconds, 1);
  assert.match(dated.summary, /^picture: differs — .*; worst frame 4, reference 1 s away \(frame differs ssim 0\.\d+ psnr \d+\.\d dB\)$/);

  // No stated time: the age is unknown, nothing differs, and the check could not tell.
  const undated = await analyseVideo(video, options);
  for (const frame of undated.frames.filter((entry) => entry.timeSeconds < 5)) {
    assert.equal(frame.crops.frame.verdict, 'unlike the reference frame');
    assert.equal(frame.crops.frame.referenceAgeSeconds, null);
    assert.match(frame.crops.frame.reasons[0], /a reference of unknown age/);
  }
  assert.equal(undated.verdict, 'unlike the reference frame');
  assert.ok(COULD_NOT_CHECK.includes('unlike the reference frame') && !FINDINGS.includes('unlike the reference frame'));
  assert.equal(exitCodeFor(undated.verdict), 3);

  // The command line: the limit has a flag; at half a second the frame at 4 s is too old, so nothing is found.
  const narrow = check(dir, video, ['--reference', ref, '--reference-at', '5', '--reference-max-age', '0.5', '--max-width', '160', '--freeze-seconds', String(NO_FREEZE)]);
  assert.equal(narrow.json?.thresholds.referenceMaxAgeSeconds, 0.5, narrow.stderr);
  assert.equal(narrow.json.verdict, 'unlike the reference frame');
  assert.equal(narrow.status, 3);

  // An image has no time of its own: its reference's age is stated, and --reference-at belongs to a recording.
  const base = pngFile(dir, 'grad.png', gradient(160, 90));
  const dim = pngFile(dir, 'dim.png', grey(160, 90, 48));
  assert.equal((await analyseImage(dim, { reference: { file: base } })).verdict, 'unlike the reference frame');
  assert.equal((await analyseImage(dim, { reference: { file: base }, referenceAgeSeconds: 0 })).verdict, 'differs');
  assert.equal((await analyseImage(dim, { reference: { file: base }, referenceAgeSeconds: 10 })).verdict, 'unlike the reference frame');
  const stated = check(dir, dim, ['--reference', base, '--reference-age', '1']);
  assert.equal(stated.json?.verdict, 'differs', stated.stderr);
  assert.equal(stated.status, 1);
  assert.equal(check(dir, dim, ['--reference', base, '--reference-at', '1']).status, 2);
  assert.equal(check(dir, video, ['--reference', ref, '--reference-age', '1']).status, 2);
});

// The tool check of 2026-11-26 (Picture): `clean` over a 1.3% black hole, whose whole-frame SSIM stays 0.99. Each
// crop's dark share (blackframe's percent under the threshold) is now compared with the same crop of the
// reference: a rise of `blackRise` points is black where the crop otherwise matches the reference or the
// reference is fresh, so a black object is found by the crop it sits in. Against an old reference of another view
// the real recording's shadows raised the whole frame's share 1 to 3 points, so that rise alone is not a finding.
test('a crop whose dark share rises over the reference crop is black: a 1.3% hole in the frame, a box in a crop', async (t) => {
  const dir = tempDir(t);
  const grad = pngFile(dir, 'grad.png', gradient(640, 360));
  const rect = pngFile(dir, 'rect.png', painted(gradient(640, 360), box(100, 100, 64, 48, black())));
  const flag = pngFile(dir, 'flag.png', painted(gradient(640, 360), box(400, 60, 40, 40, black())));
  const reference = { file: grad };

  const hole = await analyseImage(rect, { reference });
  assert.equal(crop(hole).verdict, 'black', JSON.stringify(crop(hole)));
  assert.equal(crop(hole).blackShare, 0.01);
  assert.equal(crop(hole).referenceBlackShare, 0);
  assert.ok(crop(hole).ssim > DEFAULTS.ssimMin, 'SSIM alone passes it');
  assert.match(crop(hole).reasons.join('; '), /blackframe: 1% under luma 24, the reference 0%/);
  assert.equal(hole.verdict, 'black');
  assert.equal(hole.thresholds.blackRise, 1);

  // A box covering a quarter of a crop: black against a fresh reference. Against an unaged one the crop matches it
  // nowhere else (SSIM about 0.5), so it may be another view: unlike the reference frame, with its dark shares.
  const boxed = await analyseImage(flag, { crops: FLAG_CROP, reference, referenceAgeSeconds: 0 });
  assert.equal(crop(boxed, 'flag').verdict, 'black');
  assert.equal(crop(boxed, 'flag').blackShare, 0.25);
  assert.equal(crop(boxed, 'flag').referenceBlackShare, 0);
  const unaged = crop(await analyseImage(flag, { crops: FLAG_CROP, reference }), 'flag');
  assert.equal(unaged.verdict, 'unlike the reference frame');
  assert.equal(unaged.blackShare, 0.25);
  assert.match(unaged.reasons[0], /blackframe: 25% under luma 24, the reference 0%/);
  // A crop all black needs no reference's age: 98% of it under the threshold is black on its own.
  const filled = pngFile(dir, 'filled.png', painted(gradient(640, 360), box(380, 40, 80, 80, black())));
  assert.equal(crop(await analyseImage(filled, { crops: FLAG_CROP, reference }), 'flag').verdict, 'black');

  // Dark in the reference too is clean.
  assert.equal((await analyseImage(rect, { reference: { file: rect } })).verdict, 'clean');
  assert.equal((await analyseImage(flag, { crops: FLAG_CROP, reference: { file: flag } })).verdict, 'clean');

  // The rise has a flag: at 2 points the whole frame's 1% no longer counts.
  const raised = check(dir, rect, ['--reference', grad, '--black-rise', '2']);
  assert.equal(raised.json?.thresholds.blackRise, 2, raised.stderr);
  assert.equal(raised.json.verdict, 'clean');
  assert.equal(raised.status, 0);
});

test('a numbers-only reference is refused, and a threshold flag reaches the filter options in the output', (t) => {
  const dir = tempDir(t);
  const frameFile = pngFile(dir, 'grey.png', grey(64, 32));
  const old = path.join(dir, 'reference.json');
  fs.writeFileSync(old, JSON.stringify({ version: 2, crops: {} }));
  const refused = check(dir, frameFile, ['--reference', old]);
  assert.equal(refused.status, 3);
  assert.match(refused.stderr, /make it with the reference command/);
  // A missing reference.json means the frame beside it.
  const beside = pngFile(path.join(dir), 'reference.png', grey(64, 32));
  fs.rmSync(old);
  assert.deepEqual(loadReference(old), { file: path.resolve(beside) });

  const flagged = check(dir, frameFile, ['--reference', beside, '--black-threshold', '40', '--ssim-min', '0.95']);
  assert.equal(flagged.status, 0, flagged.stderr);
  assert.equal(flagged.json.thresholds.blackThreshold, 40);
  assert.equal(flagged.json.thresholds.ssimMin, 0.95);
  assert.equal(flagged.json.filters.blackframe, 'blackframe=amount=0:threshold=40');
});
