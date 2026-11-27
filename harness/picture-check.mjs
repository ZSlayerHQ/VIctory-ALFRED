#!/usr/bin/env node
// Check a frame or a recording with ffmpeg's own detectors, on named crops, against a reference frame of the same
// view, and report their output as numbers and verdicts.
//
//   node harness/picture-check.mjs check <frame.png | recording.mp4> [--crops <json>] [--reference <image>]
//        [--reference-at <s> | --reference-age <s>] [--from <s>] [--to <s>] [--step <s>] [--max-width <px>]
//        [--json] [threshold flags]
//   node harness/picture-check.mjs reference --frame <image> [--out <png>]
//   node harness/picture-check.mjs reference --video <recording.mp4> --at <s> [--out <png>]
//
// What it can claim. From the recording alone: `black` (a crop at least `--black-amount` under the black threshold)
// and `frozen` (the whole frame unchanged). `differs` needs a fresh reference of the same view: a live scene moves,
// and the real recording's good frames read SSIM 0.60 to 0.84 in crops two seconds from their own reference and
// 0.48 whole-frame thirty seconds away (check-tools-truth.md, Picture; rechecked 2026-11-26). So a crop differs
// only against a reference taken within `--reference-max-age` seconds of the frame: `--reference-at` says when in
// the checked recording it was taken, `--reference-age` states it for an image. Against an older reference, or
// one of unknown age, a crop under its SSIM limit is `unlike the reference frame`: its figures are reported and it
// is could-not-check, never a finding.
//
// The detectors are ffmpeg's, not ours (established tools are the judges). Our code picks the frames (one every `--step`
// seconds of the window) and the crops, builds one filter graph, and parses the filters' frame metadata, which
// `metadata=mode=print` writes per frame:
//   blackframe=amount=0:threshold=T  `lavfi.blackframe.pblack`: the whole percent of the crop's pixels, and of the
//                                    reference's same crop, whose luma is under T. The recording is limited-range,
//                                    so black is luma 16.
//   freezedetect=n=N:d=D             on the whole frame: the mean absolute change of all planes, as a fraction of
//                                    full scale, stays under N for D seconds. An open freeze runs to the end.
//   mpdecimate=hi=H:lo=L:frac=F      on the whole frame: a sample repeats the last kept one when no 8x8 block's
//                                    summed difference passes H and under F of the blocks pass L.
//   ssim, psnr                       per crop, against the same crop of the reference frame (scaled to the
//                                    frames' size): `lavfi.ssim.All`, `lavfi.ssim.dB`, `lavfi.psnr.psnr_avg`,
//                                    `lavfi.psnr.mse_avg`.
// Every option passed is in the output (`filters`, `thresholds`), and each has a flag.
//
// A crops file names rectangles in fractions of the frame, so a crop survives a resolution change:
//
//   {"crops": [{"name": "shirt-seam", "x": 0.40, "y": 0.55, "w": 0.12, "h": 0.08}]}
//
// Missing, the whole frame is one crop named `frame`. The crops file and the reference frame (`reference.png`,
// made by the `reference` command from a good frame of the same view) live in the captures root's `reference/`
// folder; the root comes from the harness's configuration (VICTORY_CAPTURES or harness/local.env), never from a
// path written here. Pixels stay on this PC: the output is numbers and verdicts only, never pixel data or a path.
//
// A crop's verdict, per crop and per frame, with its SSIM, PSNR and the reference's age (`referenceAgeSeconds`,
// null when unknown) beside it:
//   black       blackframe puts at least `--black-amount` percent of it under the threshold, or at least
//               `--black-rise` points more of it than of the reference's crop where that reference is fresh or the
//               crop otherwise matches it (SSIM at its limit: the same view); a reference crop itself at
//               `--black-amount` makes it clean. Against an old reference of another view, dark shares rose 1 to 3
//               points on good frames;
//   differs     SSIM under its `ssimMin` against a reference at most `--reference-max-age` seconds from the frame;
//   unlike the reference frame  the same against an older or unaged reference (could-not-check), a dark-share
//               rise included;
//   clean       compared, and none of these; `unchecked` with no reference and not black.
// A frame is `frozen` when freezedetect and mpdecimate both find the whole frame unchanged for `--freeze-seconds`.
// freezedetect alone is a mean: an 8 px box crossing 640x360 moves it 0.0001 to 0.0003, as much as libx264's noise
// on a still textured frame, so it read such a mover as frozen; mpdecimate's blocks see the box.
// The whole check is `clean` only when every crop of every frame was compared and nothing was found; otherwise a
// finding (`differs`, `black`, `frozen`) or a could-not-check verdict:
//   unchecked   no reference, and neither blackframe nor the freeze detectors found anything;
//   unlike the reference frame  a crop under its SSIM limit against an old or unaged reference;
//   no frames   no frame fell inside the window (for example a window past the recording's end);
//   incomplete  the recording ends before the window does: it is still being written (its size changed while it
//               was read, or it ends inside a fragment) or was cut short. A finding in the part read still stands.
//               A recording that ends on a whole fragment and does not grow during the read looks finished;
//   error       (stored by annotatePictureSegments only) the recording could not be read.
// An input that cannot be read (junk, empty, missing) is an error: no verdict, a message without a path.
//
// Limits: blackframe counts whole percents, so a rise under one point (a hole under 1% of its crop) is not seen and
// a share on a percent's edge can tip by one: a fault that matters gets a crop of its own. A dark-but-wrong surface
// (a black flag at luma 30) is over the black threshold and shows only as differs, so only against a fresh
// reference. A 4 px mover in 1920x1080 measured 960 wide reads frozen.
//
// Exit codes (`check`): 0 clean; 1 findings; 2 bad arguments; 3 could not check (unchecked, unlike the reference
// frame, no frames, incomplete, or an unreadable input). `reference`: 0 written; 2 bad arguments; 3 could not read
// the frame.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { loadLocalEnv } from './local-env.mjs';
import { FRAME_RE } from './parse-run.mjs';
import { captureFolderFromLog, capturesRoot } from './presentmon.mjs';

// Version 3: ffmpeg's detectors against a reference frame. A numbers-only reference (versions 1 and 2) is refused.
export const VERSION = 3;

// blackThreshold 24: blackframe's luma threshold. Black is 16 in the recording's limited range; 24 leaves 8 levels
//   of encoder noise and stays under a night scene (luma 28 and up), which must not read as black.
// blackAmount 98: blackframe's percentage for a black crop on its own (ffmpeg's own default `amount`).
// blackRise 1: points of a crop's dark share over the reference crop's that make it black: the least blackframe
//   reports, so a 64x48 hole in 640x360 (1.3%) is black in the whole frame.
// ssimMin 0.90: a crop whose SSIM (All) against a fresh reference is under this differs. Small faults need a crop
//   around them: a 40x40 box changes a whole 640x360 frame's SSIM by about one percent.
// referenceMaxAgeSeconds 1: the oldest reference, in seconds from the frame, a crop may differ from; the real
//   recording's good frames read 1.00 in every crop at 1 s and 0.60 to 0.84 at 2 s (one scene).
// freezeNoise 0.001, freezeSeconds 5: freezedetect's `n` (ffmpeg's default) and `d`, which mpdecimate shares.
// decimateHi 768, decimateLo 320, decimateFrac 0.33: mpdecimate's defaults (an 8x8 block's summed difference: 768
//   is 12 levels a texel); measured, a still textured frame through libx264 keeps 1 or 2 of 12 samples, an 8 px
//   mover all of them.
export const DEFAULTS = Object.freeze({
  blackThreshold: 24,
  blackAmount: 98,
  blackRise: 1,
  ssimMin: 0.9,
  referenceMaxAgeSeconds: 1,
  freezeNoise: 0.001,
  freezeSeconds: 5,
  decimateHi: 768,
  decimateLo: 320,
  decimateFrac: 0.33,
  stepSeconds: 1,
  maxWidth: 1920,
});
const THRESHOLD_KEYS = [
  'blackThreshold',
  'blackAmount',
  'blackRise',
  'ssimMin',
  'referenceMaxAgeSeconds',
  'freezeNoise',
  'freezeSeconds',
  'decimateHi',
  'decimateLo',
  'decimateFrac',
  'stepSeconds',
];
const thresholdsOf = (opts) => Object.fromEntries(THRESHOLD_KEYS.map((key) => [key, opts[key]]));

export const UNLIKE = 'unlike the reference frame';
export const FINDINGS = Object.freeze(['differs', 'black', 'frozen']);
export const COULD_NOT_CHECK = Object.freeze(['unchecked', UNLIKE, 'no frames', 'incomplete', 'error']);
export const VERDICTS = Object.freeze(['clean', ...FINDINGS, ...COULD_NOT_CHECK]);
// Within a frame and across frames: any finding beats could-not-check (`unlike`, which has figures, over
// `unchecked`), which beats `clean`.
const RANK = { clean: 0, unchecked: 0.5, [UNLIKE]: 0.6, differs: 1, black: 2, frozen: 3 };
const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$/;

/// The command's exit code for a check's verdict: 0 clean, 1 a finding, 3 could not check.
export function exitCodeFor(verdict) {
  if (verdict === 'clean') return 0;
  if (FINDINGS.includes(verdict)) return 1;
  return 3;
}

const round = (value, digits) => {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
};
const points = (value) => `${value} point${value === 1 ? '' : 's'}`;
const pct = (value) => `${(value * 100).toFixed(0)}%`;
const firstLine = (text) => String(text || '').trim().split(/\r?\n/)[0] || 'no error';
// An error says what went wrong, never a folder path of this machine: output may be shared.
const safeMessage = (err) => {
  const text = String(err?.message || 'failed');
  if (/[\\/]/.test(text) || /^(?:ENOENT|EACCES|EPERM|ENOTDIR)\b/.test(text)) {
    return err?.code || 'the file could not be read';
  }
  return text;
};

// ----------------------------------------------------------------------------------------------------------------
// crops and the reference

/// The whole frame under one crop name, for a run without a crops file.
export function frameCrop() {
  return [{ name: 'frame', x: 0, y: 0, w: 1, h: 1 }];
}

function cropList(input) {
  if (Array.isArray(input)) return input;
  if (!input || typeof input !== 'object') return null;
  if (Array.isArray(input.crops)) return input.crops;
  if (typeof input.name === 'string' && 'x' in input) return [input];
  if (input.crops && typeof input.crops === 'object') {
    return Object.entries(input.crops).map(([name, crop]) => ({ name, ...crop }));
  }
  const values = Object.values(input);
  if (values.length && values.every((crop) => crop && typeof crop === 'object' && !Array.isArray(crop) && 'x' in crop)) {
    return Object.entries(input).map(([name, crop]) => ({ name, ...crop }));
  }
  return null;
}

/// A crops JSON value as a list: an array, `{ "crops": [...] }`, `{ "crops": { name: { x, y, w, h } } }`, a map of
/// names to rectangles, or one crop object. Every rectangle is a fraction of the frame (0 to 1) and every name is a
/// plain label, so no name can carry a path. A crop may set its own `ssimMin` (a tight crop around a seam of
/// single dark texels needs one near 1: the default 0.9 does not see sparse specks).
export function normalizeCrops(input) {
  const list = cropList(input);
  if (!list) throw new Error('the crops file needs a "crops" array (or a list of crops)');
  if (!list.length) throw new Error('the crops file has no crops');
  const names = new Set();
  return list.map((crop, index) => {
    const where = `crop ${index + 1}`;
    if (!crop || typeof crop !== 'object' || Array.isArray(crop)) throw new Error(`${where} is not an object`);
    if (typeof crop.name !== 'string' || !NAME_RE.test(crop.name)) {
      throw new Error(`${where} needs a name of letters, digits, spaces, dots, dashes or underscores`);
    }
    if (names.has(crop.name)) throw new Error(`${where}: duplicate name '${crop.name}'`);
    names.add(crop.name);
    for (const key of ['x', 'y', 'w', 'h']) {
      if (typeof crop[key] !== 'number' || !Number.isFinite(crop[key])) {
        throw new Error(`${where} ('${crop.name}'): ${key} must be a number`);
      }
    }
    if (crop.x < 0 || crop.y < 0 || crop.w <= 0 || crop.h <= 0) {
      throw new Error(`${where} ('${crop.name}'): x and y are 0 or more; w and h are over 0`);
    }
    if (crop.x + crop.w > 1 + 1e-9 || crop.y + crop.h > 1 + 1e-9) {
      throw new Error(`${where} ('${crop.name}'): the rectangle reaches past the frame`);
    }
    if (crop.ssimMin !== undefined && (typeof crop.ssimMin !== 'number' || crop.ssimMin < 0 || crop.ssimMin > 1)) {
      throw new Error(`${where} ('${crop.name}'): ssimMin is a number from 0 to 1`);
    }
    return { name: crop.name, x: crop.x, y: crop.y, w: crop.w, h: crop.h, ...(crop.ssimMin === undefined ? {} : { ssimMin: crop.ssimMin }) };
  });
}

/// A crops file's crops; a missing file is the whole frame, named `frame`. A malformed file is refused, not
/// silently turned into the whole frame.
export function loadCrops(file) {
  if (!file) return frameCrop();
  let text;
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') return frameCrop();
    throw err;
  }
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error('the crops file is not JSON');
  }
  return normalizeCrops(parsed);
}

/// The reference frame: `{ file }` for an image of the same view, or null when there is none. A `.json` path is the
/// old numbers-only reference: one that exists is refused (make a frame with the `reference` command); a missing
/// one means the `reference.png` beside it, so a caller that names the old file finds the new one.
export function loadReference(file) {
  if (!file) return null;
  if (/\.json$/i.test(file)) {
    if (fs.existsSync(file)) {
      throw new Error('a reference is now a frame of the same view (reference.png), not numbers: make it with the reference command');
    }
    const beside = path.join(path.dirname(file), 'reference.png');
    return fs.existsSync(beside) ? { file: path.resolve(beside) } : null;
  }
  return fs.existsSync(file) ? { file: path.resolve(file) } : null;
}

function referenceOf(reference) {
  if (!reference) return null;
  if (typeof reference === 'string') return loadReference(reference);
  if (typeof reference.file === 'string') return { file: path.resolve(reference.file) };
  throw new Error('the reference is a frame image ({ file }), not numbers');
}

/// The crop's pixel rectangle in a frame. Fractions are rounded at each edge and clamped to the frame, so the same
/// crop keeps its contents when the frame is resized.
export function cropRect(crop, width, height) {
  const left = Math.max(0, Math.min(width - 1, Math.round(crop.x * width)));
  const top = Math.max(0, Math.min(height - 1, Math.round(crop.y * height)));
  const right = Math.max(left + 1, Math.min(width, Math.round((crop.x + crop.w) * width)));
  const bottom = Math.max(top + 1, Math.min(height, Math.round((crop.y + crop.h) * height)));
  return { x: left, y: top, w: right - left, h: bottom - top };
}

// The rectangle on the even grid that yuv420p's crop needs, at least 16x16: SSIM's window is 8x8 and the chroma
// planes are half size, so a smaller crop gives no SSIM for them.
function filterRect(crop, width, height) {
  const rect = cropRect(crop, width, height);
  // width and height are even (analysisSize), so every span below stays even.
  const axis = (start, length, size) => {
    let from = start - (start % 2);
    let span = start + length - from;
    span = Math.max(16, span + (span % 2));
    if (from + span > size) from = Math.max(0, size - span);
    from -= from % 2;
    span = Math.min(span, size - from);
    return [from, span - (span % 2)];
  };
  const [x, w] = axis(rect.x, rect.w, width);
  const [y, h] = axis(rect.y, rect.h, height);
  return { x, y, w, h };
}

// ----------------------------------------------------------------------------------------------------------------
// ffmpeg

/// A file's size as ffprobe sees it. ffprobe comes with the ffmpeg the harness already uses.
export function probeVideoSize(video, ffprobe = 'ffprobe') {
  const result = spawnSync(
    ffprobe,
    ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video],
    { encoding: 'utf8' },
  );
  if (result.status !== 0) throw new Error(`ffprobe failed: ${firstLine(result.stderr)}`);
  const match = (result.stdout || '').trim().match(/^(\d+)x(\d+)/);
  if (!match) throw new Error("ffprobe did not report the input's size");
  return { width: Number(match[1]), height: Number(match[2]) };
}

// The size the frames are measured at: the input's, no wider than `maxWidth` or the reference frame (so neither is
// scaled up, which blurs edges SSIM then counts), on the even grid yuv420p needs.
function analysisSize(native, maxWidth, referenceWidth = null) {
  let width = native.width;
  let height = native.height;
  const limit = Math.min(maxWidth || Infinity, referenceWidth || Infinity);
  if (width > limit) {
    width = limit;
    height = Math.round((native.height * width) / native.width);
  }
  const even = (value) => Math.max(16, value - (value % 2));
  return { width: even(width), height: even(height) };
}

/// The filters and their options, as passed to ffmpeg and printed in the output.
export function filterOptions(opts) {
  return {
    blackframe: `blackframe=amount=0:threshold=${opts.blackThreshold}`,
    freezedetect: `freezedetect=n=${opts.freezeNoise}:d=${opts.freezeSeconds}`,
    mpdecimate: `mpdecimate=hi=${opts.decimateHi}:lo=${opts.decimateLo}:frac=${opts.decimateFrac}`,
    ssim: 'ssim',
    psnr: 'psnr',
  };
}

// The graph's output labels, in the order `runFilters` maps them.
function graphOutputs(crops, { sample }) {
  return ['owhole', ...(sample ? ['okept'] : []), ...crops.map((_, i) => `o${i}`)];
}

/// One filter graph for the whole frame and every crop. `[owhole]` carries blackframe and freezedetect on the whole
/// frame and, for a recording, `[okept]` the samples mpdecimate keeps (tagged, since a print skips a frame without
/// metadata); `[o<i>]` carries crop i's blackframe and, with a reference, its SSIM and PSNR against the same crop of
/// the reference. Each branch ends in a metadata print to a file named for it, in ffmpeg's working folder.
export function filterGraph({ width, height, crops, reference, sample, opts }) {
  const filters = filterOptions(opts);
  const parts = [];
  const head = [sample ? (sample === 1 ? 'fps=1' : `fps=1/${sample}`) : null, `scale=${width}:${height}`, 'format=yuv420p'];
  const kept = sample ? '[kept]' : '';
  parts.push(`[0:v]${head.filter(Boolean).join(',')},split=${crops.length + (sample ? 2 : 1)}[whole]${kept}${crops.map((_, i) => `[m${i}]`).join('')}`);
  parts.push(`[whole]${filters.blackframe},${filters.freezedetect},metadata=mode=print:file=whole.txt[owhole]`);
  if (sample) {
    parts.push(`[kept]${filters.mpdecimate},metadata=mode=add:key=picture.kept:value=1,metadata=mode=print:file=kept.txt[okept]`);
  }
  if (reference) {
    const labels = crops.flatMap((_, i) => [`[ra${i}]`, `[rb${i}]`]).join('');
    parts.push(`[1:v]scale=${width}:${height},format=yuv420p,split=${crops.length * 2}${labels}`);
  }
  crops.forEach((crop, i) => {
    const rect = filterRect(crop, width, height);
    const cut = `crop=${rect.w}:${rect.h}:${rect.x}:${rect.y}`;
    if (reference) {
      parts.push(`[ra${i}]${cut}[rca${i}]`, `[rb${i}]${cut}[rcb${i}]`, `[m${i}]${cut}[c${i}]`);
      parts.push(`[c${i}][rca${i}]${filters.ssim}[s${i}]`);
      parts.push(`[s${i}][rcb${i}]${filters.psnr},${filters.blackframe},metadata=mode=print:file=crop-${i}.txt[o${i}]`);
    } else {
      parts.push(`[m${i}]${cut},${filters.blackframe},metadata=mode=print:file=crop-${i}.txt[o${i}]`);
    }
  });
  return parts.join(';');
}

/// The reference frame's own pass: each crop's blackframe at the frames' size, printed to `reference-<i>.txt` and
/// mapped as `[or<i>]`. Apart from the main graph, whose empty streams (a window past the end) would fail to open
/// beside the reference's one frame.
export function referenceGraph({ width, height, crops, opts }) {
  const { blackframe } = filterOptions(opts);
  const parts = [`[0:v]scale=${width}:${height},format=yuv420p,split=${crops.length}${crops.map((_, i) => `[r${i}]`).join('')}`];
  crops.forEach((crop, i) => {
    const rect = filterRect(crop, width, height);
    parts.push(`[r${i}]crop=${rect.w}:${rect.h}:${rect.x}:${rect.y},${blackframe},metadata=mode=print:file=reference-${i}.txt[or${i}]`);
  });
  return parts.join(';');
}

/// ffmpeg's `metadata=mode=print` output as frames: `{ frame, ptsTime, meta: { key: value } }`.
export function parseMetadataPrint(text) {
  const frames = [];
  let current = null;
  for (const line of String(text).split(/\r?\n/)) {
    const head = line.match(/^frame:\s*(\d+)\s+pts:\s*\S+\s+pts_time:\s*(\S+)/);
    if (head) {
      current = { frame: Number(head[1]), ptsTime: Number(head[2]), meta: {} };
      frames.push(current);
      continue;
    }
    const pair = line.match(/^([^=\s]+)=(.*)$/);
    if (pair && current) current.meta[pair[1]] = pair[2].trim();
  }
  return frames;
}

function runProcess(command, args, options) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { ...options, stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    child.stderr.on('data', (chunk) => {
      stderr += chunk;
    });
    child.on('error', reject);
    child.on('close', (code) => resolve({ code, stderr }));
  });
}

/// One ffmpeg pass over an image or a recording range. Returns the parsed whole-frame and per-crop metadata, the
/// size measured at, and `growing` (the file's size or time changed during the read; `statFile` is a seam for the
/// tests) and `partial` (ffmpeg met a fragment cut short at the file's end).
export async function runFilters(input, { crops, reference, fromSeconds = 0, toSeconds = null, image = false, ...options } = {}) {
  const opts = { ...DEFAULTS, ...options };
  const file = path.resolve(input);
  const ref = referenceOf(reference);
  const statFile = opts.statFile ?? ((name) => fs.statSync(name));
  const before = statFile(file);
  const ffprobe = opts.ffprobe ?? 'ffprobe';
  const size = analysisSize(probeVideoSize(file, ffprobe), opts.maxWidth, ref ? probeVideoSize(ref.file, ffprobe).width : null);
  const sample = image ? null : opts.stepSeconds;
  const graph = filterGraph({ ...size, crops, reference: ref, sample, opts });
  const args = ['-hide_banner', '-nostdin', '-v', 'error'];
  if (!image && fromSeconds > 0) args.push('-ss', String(fromSeconds));
  if (!image && toSeconds !== null && Number.isFinite(toSeconds) && toSeconds > fromSeconds) {
    args.push('-t', String(toSeconds - fromSeconds));
  }
  args.push('-i', file);
  if (ref) args.push('-i', ref.file);
  args.push('-filter_complex', graph);
  for (const label of graphOutputs(crops, { sample })) args.push('-map', `[${label}]`);
  args.push('-f', 'null', '-');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'picture-check-'));
  try {
    const { code, stderr } = await runProcess(opts.ffmpeg ?? 'ffmpeg', args, { cwd: dir });
    if (code !== 0) throw new Error(`ffmpeg failed: ${firstLine(stderr)}`);
    if (ref) {
      const refArgs = ['-hide_banner', '-nostdin', '-v', 'error', '-i', ref.file, '-filter_complex', referenceGraph({ ...size, crops, opts })];
      crops.forEach((_, i) => refArgs.push('-map', `[or${i}]`));
      refArgs.push('-f', 'null', '-');
      const pass = await runProcess(opts.ffmpeg ?? 'ffmpeg', refArgs, { cwd: dir });
      if (pass.code !== 0) throw new Error(`ffmpeg failed on the reference: ${firstLine(pass.stderr)}`);
    }
    const read = (name) => (fs.existsSync(path.join(dir, name)) ? fs.readFileSync(path.join(dir, name), 'utf8') : '');
    const after = statFile(file);
    return {
      ...size,
      whole: parseMetadataPrint(read('whole.txt')),
      kept: sample ? parseMetadataPrint(read('kept.txt')) : null,
      crops: crops.map((_, i) => parseMetadataPrint(read(`crop-${i}.txt`))),
      // The reference is one frame: its crop's metadata, or null without a reference.
      referenceCrops: crops.map((_, i) => (ref ? (parseMetadataPrint(read(`reference-${i}.txt`))[0]?.meta ?? {}) : null)),
      growing: before.size !== after.size || before.mtimeMs !== after.mtimeMs,
      partial: /partial file/i.test(stderr),
    };
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

// ----------------------------------------------------------------------------------------------------------------
// verdicts

const numberOf = (value) => {
  if (value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

// A crop's figures and verdict (the header's rules). `referenceMeta` is the reference crop's blackframe, or null
// without a reference; `ageSeconds` is the reference's distance from this frame, or null when unknown.
function cropEntry(meta, referenceMeta, ageSeconds, opts, crop = {}) {
  const hasReference = referenceMeta !== null;
  const ssimMin = crop.ssimMin ?? opts.ssimMin;
  const pblack = numberOf(meta['lavfi.blackframe.pblack']);
  const referencePblack = hasReference ? numberOf(referenceMeta['lavfi.blackframe.pblack']) : null;
  const entry = {
    blackShare: pblack === null ? null : pblack / 100,
    referenceBlackShare: referencePblack === null ? null : referencePblack / 100,
    ssim: numberOf(meta['lavfi.ssim.All']),
    ssimY: numberOf(meta['lavfi.ssim.Y']),
    ssimDb: numberOf(meta['lavfi.ssim.dB']),
    // Identical crops give an infinite PSNR, which ffmpeg prints as inf: null here, with a mean square error of 0.
    psnr: numberOf(meta['lavfi.psnr.psnr_avg']),
    mse: numberOf(meta['lavfi.psnr.mse_avg']),
    referenceAgeSeconds: hasReference ? ageSeconds : null,
  };
  const allBlack = pblack !== null && pblack >= opts.blackAmount;
  const blackReason = hasReference
    ? `blackframe: ${pblack}% under luma ${opts.blackThreshold}, the reference ${referencePblack ?? 'n/a'}% ` +
      `(black from ${opts.blackAmount}% or ${points(opts.blackRise)} over the reference)`
    : `blackframe: ${pblack}% under luma ${opts.blackThreshold} (black from ${opts.blackAmount}%)`;
  if (!hasReference) {
    return allBlack
      ? { ...entry, verdict: 'black', reasons: [blackReason, 'no reference'] }
      : { ...entry, verdict: 'unchecked', reasons: ['no reference'] };
  }
  // A rise over the reference's dark share counts where the reference is fresh or the crop otherwise matches it:
  // against an old reference of another view, shadows alone raise it.
  const fresh = ageSeconds !== null && ageSeconds <= opts.referenceMaxAgeSeconds;
  const matches = entry.ssim !== null && entry.ssim >= ssimMin;
  const referenceBlack = referencePblack !== null && referencePblack >= opts.blackAmount;
  const rise = pblack !== null && referencePblack !== null && pblack - referencePblack >= opts.blackRise;
  const black = !referenceBlack && (allBlack || (rise && (fresh || matches)));
  const reasons = black || (rise && !referenceBlack) ? [blackReason] : [];
  if (entry.ssim === null) {
    return black ? { ...entry, verdict: 'black', reasons } : { ...entry, verdict: 'unchecked', reasons: ['ssim gave no value'] };
  }
  if (matches) return { ...entry, verdict: black ? 'black' : 'clean', reasons };
  const against = ageSeconds === null ? 'a reference of unknown age' : `a reference ${ageSeconds} s away`;
  const limit = fresh ? '' : `; differs needs one within ${opts.referenceMaxAgeSeconds} s`;
  reasons.push(`ssim ${entry.ssim} against ${against}, under ${ssimMin}${limit}`);
  return { ...entry, verdict: black ? 'black' : fresh ? 'differs' : UNLIKE, reasons };
}

const worstOf = (verdicts) =>
  verdicts.reduce((worst, verdict) => (RANK[verdict] > RANK[worst] ? verdict : worst), 'clean');

// freezedetect's intervals in recording seconds. A freeze still open at the last frame runs past it.
function freezeIntervals(whole, fromSeconds, stepSeconds) {
  const intervals = [];
  let open = null;
  for (const frame of whole) {
    const start = numberOf(frame.meta['lavfi.freezedetect.freeze_start']);
    const end = numberOf(frame.meta['lavfi.freezedetect.freeze_end']);
    if (end !== null && open) {
      intervals.push({ fromSeconds: open, toSeconds: fromSeconds + end });
      open = null;
    }
    if (start !== null) open = fromSeconds + start;
  }
  if (open !== null) {
    const last = whole.length ? fromSeconds + whole[whole.length - 1].ptsTime : open;
    intervals.push({ fromSeconds: open, toSeconds: last + stepSeconds, open: true });
  }
  return intervals;
}

// mpdecimate's still runs in recording seconds: a kept sample and the samples dropped after it as its repeats.
// Only a run of at least `freezeSeconds` counts, as freezedetect's `d`.
function stillRuns(whole, kept, fromSeconds, freezeSeconds) {
  if (!kept) return [];
  const keptTimes = new Set(kept.map((frame) => frame.ptsTime));
  const runs = [];
  let run = null;
  for (const frame of whole) {
    const at = round(fromSeconds + frame.ptsTime, 3);
    if (!run || keptTimes.has(frame.ptsTime)) {
      if (run) runs.push(run);
      run = { fromSeconds: at, toSeconds: at };
    } else {
      run.toSeconds = at;
    }
  }
  if (run) runs.push(run);
  return runs.filter((entry) => entry.toSeconds - entry.fromSeconds >= freezeSeconds);
}

/// The worst entry per crop over a run's frames, for a summary's one line per crop: by verdict, then the lowest SSIM.
export function cropVerdicts(frames) {
  const out = {};
  for (const frame of frames) {
    for (const [name, crop] of Object.entries(frame.crops)) {
      const held = out[name];
      const lower = (crop.ssim ?? 2) < (held?.ssim ?? 2);
      if (!held || RANK[crop.verdict] > RANK[held.verdict] || (RANK[crop.verdict] === RANK[held.verdict] && lower)) {
        out[name] = { ...crop };
      }
    }
  }
  return out;
}

function pickWorst(frames) {
  let worst = null;
  for (const entry of frames) {
    const ssim = Math.min(...Object.values(entry.crops).map((crop) => crop.ssim ?? 2));
    if (!worst || RANK[entry.verdict] > RANK[worst.entry.verdict] || (RANK[entry.verdict] === RANK[worst.entry.verdict] && ssim < worst.ssim)) {
      worst = { entry, ssim };
    }
  }
  return worst ? { ...worst.entry, crops: Object.fromEntries(Object.entries(worst.entry.crops).map(([k, v]) => [k, { ...v }])) } : null;
}

function summaryOf({ frames, verdict, frozen, worstFrame, crops, hasReference }) {
  if (!frames.length) return 'picture: no frames';
  const count = `${frames.length} frame${frames.length === 1 ? '' : 's'}`;
  const frozenText = frozen.verdict === 'frozen' ? `, frozen ${frozen.seconds.toFixed(1)} s` : '';
  const cropText = crops
    .map((crop) => {
      const entry = worstFrame?.crops[crop.name];
      if (!entry) return `${crop.name} unchecked`;
      const ssim = entry.ssim === null ? '' : ` ssim ${entry.ssim.toFixed(3)}`;
      const psnr = entry.psnr === null ? '' : ` psnr ${entry.psnr.toFixed(1)} dB`;
      return `${crop.name} ${entry.verdict}${ssim}${psnr}`;
    })
    .join(', ');
  const head = hasReference ? `picture: ${verdict}` : `picture: ${verdict}: no reference`;
  const age = worstFrame?.referenceAgeSeconds;
  const ageText = !hasReference || !worstFrame ? '' : age === null ? ', reference age unknown' : `, reference ${age} s away`;
  return `${head} — ${crops.length} crop${crops.length === 1 ? '' : 's'}, ${count}${frozenText}; worst frame ${worstFrame?.index ?? '—'}${ageText} (${cropText})`;
}

/// One window's result from a pass's parsed metadata: the frames whose time falls inside it. `ageOf(seconds)` is
/// the reference's distance from a frame at that time, or null when unknown.
function windowResult(read, window, { crops, hasReference, fromSeconds, ageOf, opts }) {
  const frames = [];
  const intervals = freezeIntervals(read.whole, fromSeconds, opts.stepSeconds);
  const runs = stillRuns(read.whole, read.kept, fromSeconds, opts.freezeSeconds);
  const inside = (list, at, closed) => list.some((span) => at >= span.fromSeconds && (closed ? at <= span.toSeconds : at < span.toSeconds));
  read.whole.forEach((whole, index) => {
    const timeSeconds = round(fromSeconds + whole.ptsTime, 3);
    if (timeSeconds < window.fromSeconds || timeSeconds > window.toSeconds) return;
    const referenceAgeSeconds = hasReference ? ageOf(timeSeconds) : null;
    const cropsOut = {};
    crops.forEach((crop, i) => {
      const meta = read.crops[i][index]?.meta ?? {};
      cropsOut[crop.name] = cropEntry(meta, hasReference ? read.referenceCrops[i] ?? {} : null, referenceAgeSeconds, opts, crop);
    });
    // Frozen only when both detectors find the whole frame unchanged: freezedetect's mean and mpdecimate's blocks.
    const frozen = inside(intervals, timeSeconds, false) && inside(runs, timeSeconds, true);
    frames.push({
      index: frames.length,
      timeSeconds,
      referenceAgeSeconds,
      measuredAt: { width: read.width, height: read.height },
      blackShare: numberOf(whole.meta['lavfi.blackframe.pblack']) === null ? null : Number(whole.meta['lavfi.blackframe.pblack']) / 100,
      crops: cropsOut,
      frozen,
      verdict: frozen ? 'frozen' : worstOf(Object.values(cropsOut).map((entry) => entry.verdict)),
    });
  });
  const frozenFrames = frames.filter((frame) => frame.frozen);
  const frozen = frozenFrames.length
    ? {
        verdict: 'frozen',
        seconds: round(frozenFrames[frozenFrames.length - 1].timeSeconds - frozenFrames[0].timeSeconds + opts.stepSeconds, 3),
        fromSeconds: frozenFrames[0].timeSeconds,
        toSeconds: frozenFrames[frozenFrames.length - 1].timeSeconds,
        frames: frozenFrames.length,
      }
    : { verdict: 'clean', seconds: 0, fromSeconds: null, toSeconds: null, frames: 0 };
  const verdict = !frames.length ? 'no frames' : worstOf(frames.map((frame) => frame.verdict));
  const worstFrame = pickWorst(frames);
  return {
    version: VERSION,
    tool: 'ffmpeg',
    filters: filterOptions(opts),
    thresholds: thresholdsOf(opts),
    frames,
    frozen,
    verdict,
    worstFrame,
    reference: hasReference,
    summary: summaryOf({ frames, verdict, frozen, worstFrame, crops, hasReference }),
  };
}

/// Why a window's result does not cover the window, or null. A window with a finite end is covered when its last
/// sample is within one and a half steps of the end; an open window (the whole recording) is covered unless the
/// recording grew during the read or ends inside a fragment.
function windowShortfall(result, window, read, stepSeconds) {
  if (!result.frames.length) return null;
  const cutShort = read.growing || read.partial;
  if (Number.isFinite(window.toSeconds)) {
    const last = result.frames[result.frames.length - 1].timeSeconds;
    if (last >= window.toSeconds - 1.5 * stepSeconds) return null;
    return cutShort
      ? 'the recording is still being written or was cut short before this window ends'
      : 'the recording ends before this window does';
  }
  if (read.growing) return 'the recording grew while it was read: it is still being written';
  if (read.partial) return 'the recording ends inside a fragment: it is still being written or was cut short';
  return null;
}

/// A result whose window the recording does not cover: `incomplete`, unless the part read holds a finding, which
/// stands with the reason beside it. Never `clean`.
function withShortfall(result, reason) {
  if (!reason) return result;
  const count = `${result.frames.length} frame${result.frames.length === 1 ? '' : 's'} read`;
  if (FINDINGS.includes(result.verdict)) return { ...result, incomplete: reason, summary: `${result.summary}; ${reason}` };
  return { ...result, verdict: 'incomplete', incomplete: reason, summary: `picture: incomplete — ${reason} (${count})` };
}

const ageOption = (value, name) => {
  if (value === undefined || value === null) return null;
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) throw new Error(`${name} is a number of seconds, 0 or more`);
  return value;
};

/// One image (a PNG or anything ffmpeg decodes) through the same filters. An image has no time of its own:
/// `referenceAgeSeconds` states how far from it the reference was taken (unknown without it).
export async function analyseImage(file, { crops = frameCrop(), reference = null, referenceAgeSeconds = null, ...options } = {}) {
  if (options.referenceAtSeconds != null) throw new Error('referenceAtSeconds is a time in a recording; an image takes referenceAgeSeconds');
  const age = ageOption(referenceAgeSeconds, 'referenceAgeSeconds');
  const opts = { ...DEFAULTS, ...options };
  const ref = referenceOf(reference);
  const read = await runFilters(file, { ...opts, crops, reference: ref, image: true });
  const context = { crops, hasReference: ref !== null, fromSeconds: 0, ageOf: () => age, opts };
  return windowResult(read, { fromSeconds: 0, toSeconds: Infinity }, context);
}

/// One recording pass for several time ranges (the plan runner's segments). Each window gets the frames that fall
/// inside it. Returns one result per window, in order. `referenceAtSeconds` is when in this recording the reference
/// frame was taken (`reference --video ... --at`); without it the reference's age is unknown.
export async function analyseVideoWindows(video, windows, { crops = frameCrop(), reference = null, referenceAtSeconds = null, ...options } = {}) {
  if (!windows.length) return [];
  if (options.referenceAgeSeconds != null) throw new Error('referenceAgeSeconds is for an image; a recording takes referenceAtSeconds');
  const at = ageOption(referenceAtSeconds, 'referenceAtSeconds');
  const opts = { ...DEFAULTS, ...options };
  const ref = referenceOf(reference);
  const ageOf = (seconds) => (at === null ? null : round(Math.abs(seconds - at), 3));
  const context = { crops, hasReference: ref !== null, ageOf, opts };
  const fromSeconds = Math.max(0, Math.min(...windows.map((window) => window.fromSeconds)));
  const ends = windows.map((window) => window.toSeconds).filter((end) => Number.isFinite(end));
  const toSeconds = ends.length && ends.length === windows.length ? Math.max(...ends) : null;
  const read = await runFilters(video, { ...opts, crops, reference: ref, fromSeconds, toSeconds });
  const results = windows.map((window) => {
    const result = windowResult(read, window, { ...context, fromSeconds });
    return withShortfall(result, windowShortfall(result, window, read, opts.stepSeconds));
  });
  // A window narrower than the sampling grid can fall between samples. Give it one pass of its own rather than a
  // silent clean: a short segment still gets a frame. A window that still has none says `no frames`.
  for (let index = 0; index < windows.length; index += 1) {
    if (results[index].frames.length) continue;
    const window = windows[index];
    const to = Number.isFinite(window.toSeconds) ? window.toSeconds : null;
    const own = await runFilters(video, { ...opts, crops, reference: ref, fromSeconds: window.fromSeconds, toSeconds: to });
    const result = windowResult(own, window, { ...context, fromSeconds: window.fromSeconds });
    results[index] = withShortfall(result, windowShortfall(result, window, own, opts.stepSeconds));
  }
  return results;
}

/// One recording range through the same filters.
export async function analyseVideo(video, options = {}) {
  const fromSeconds = options.fromSeconds ?? 0;
  const toSeconds = options.toSeconds ?? Infinity;
  const [result] = await analyseVideoWindows(video, [{ fromSeconds, toSeconds }], options);
  return result;
}

/// Write a reference frame: one frame of an image, or of a recording at `atSeconds`, as a PNG. Returns its size.
export function writeReference({ frame = null, video = null, atSeconds = 0, out, ffmpeg = 'ffmpeg', ffprobe = 'ffprobe' }) {
  const input = path.resolve(frame ?? video);
  probeVideoSize(input, ffprobe);
  fs.mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
  const args = ['-hide_banner', '-nostdin', '-v', 'error', '-y'];
  if (video && atSeconds > 0) args.push('-ss', String(atSeconds));
  args.push('-i', input, '-frames:v', '1', path.resolve(out));
  const result = spawnSync(ffmpeg, args, { encoding: 'utf8' });
  if (result.status !== 0 || !fs.existsSync(out) || fs.statSync(out).size === 0) {
    throw new Error(`ffmpeg wrote no frame: ${firstLine(result.stderr)}`);
  }
  return probeVideoSize(path.resolve(out), ffprobe);
}

// ----------------------------------------------------------------------------------------------------------------
// a log's frames in wall-clock time

const RECORDING_STARTED_RE = /^recording: started (\S+)\s*$/;

/// The recording's start and the log's numbered frame lines: { startedAt, samples: [{ frame, at, fps }] }. Used to
/// turn a live segment's frame range into the wall-clock seconds the recording uses.
export function frameClock(lines) {
  let startedAt = null;
  const samples = [];
  for (const raw of lines) {
    const line = String(raw).replace(/\r$/, '');
    if (startedAt === null) {
      const started = line.match(RECORDING_STARTED_RE);
      if (started) {
        const at = Date.parse(started[1]);
        if (!Number.isNaN(at)) startedAt = at;
        continue;
      }
    }
    const match = line.match(FRAME_RE);
    if (!match || match.groups.frame === undefined) continue;
    const at = Date.parse(match[1]);
    const frame = Number(match.groups.frame);
    const fps = Number(match.groups.fps);
    if (Number.isNaN(at) || !Number.isFinite(frame)) continue;
    samples.push({ frame, at, fps: Number.isFinite(fps) && fps > 0 ? fps : null });
  }
  samples.sort((a, b) => a.frame - b.frame);
  return { startedAt, samples };
}

/// The wall-clock time of a game frame, interpolated between the log's frame lines. Null without a start or a
/// sample.
export function timeOfFrame(clock, frame) {
  if (clock.startedAt === null || !clock.samples.length) return null;
  const samples = clock.samples;
  if (frame <= samples[0].frame) {
    const fps = samples[0].fps ?? 30;
    return samples[0].at + ((frame - samples[0].frame) * 1000) / fps;
  }
  const last = samples[samples.length - 1];
  if (frame >= last.frame) {
    const fps = last.fps ?? 30;
    return last.at + ((frame - last.frame) * 1000) / fps;
  }
  for (let i = 1; i < samples.length; i += 1) {
    const right = samples[i];
    if (frame > right.frame) continue;
    const left = samples[i - 1];
    if (right.frame === left.frame) return right.at;
    const fraction = (frame - left.frame) / (right.frame - left.frame);
    return left.at + fraction * (right.at - left.at);
  }
  return last.at;
}

/// A live segment's frame range as seconds from the recording's start. Null when the log cannot place it.
export function segmentWindow(segment, clock) {
  if (!segment || clock.startedAt === null || !clock.samples.length) return null;
  const first = timeOfFrame(clock, segment.firstFrame);
  const last = timeOfFrame(clock, segment.lastFrame);
  if (first === null || last === null) return null;
  const fps = clock.samples[clock.samples.length - 1].fps ?? 30;
  const end = timeOfFrame(clock, segment.lastFrame + 1) ?? last + 1000 / fps;
  const fromSeconds = Math.max(0, (first - clock.startedAt) / 1000);
  const toSeconds = Math.max(fromSeconds + 1 / fps, (end - clock.startedAt) / 1000);
  return { fromSeconds: round(fromSeconds, 3), toSeconds: round(toSeconds, 3) };
}

/// Fill each occurrence's `picture` from the recording, for segments the log can place in time. Returns a short
/// note for the runner's terminal. No recording means no check; no reference frame still runs blackframe and the
/// freeze detectors, and every crop is `unchecked: no reference`. The reference here is a frame of another run, of
/// unknown age, so a crop under its SSIM limit is `unlike the reference frame`, never `differs`. Only a compared
/// verdict counts as checked.
export async function annotatePictureSegments({
  logFile,
  logText,
  segments,
  occurrences,
  root = null,
  cropsFile = null,
  referenceFile = null,
  stepSeconds = 1,
  maxWidth = 960,
} = {}) {
  if (!logFile || !segments?.length || !occurrences?.length) return 'no segments';
  loadLocalEnv();
  const lines = (logText ?? fs.readFileSync(logFile, 'utf8')).split(/\r?\n/);
  const capture = captureFolderFromLog(lines, logFile);
  if (!capture) return 'no capture folder in the log';
  const video = path.join(capture, 'screen.mp4');
  if (!fs.existsSync(video)) return 'no recording';
  const base = root ?? capturesRoot();
  const cropsPath = cropsFile ?? path.join(base, 'reference', 'crops.json');
  const referencePath = referenceFile ?? path.join(base, 'reference', 'reference.png');
  const reference = loadReference(referencePath);
  const crops = loadCrops(cropsPath);
  const clock = frameClock(lines);
  if (clock.startedAt === null) return 'the log has no recording start';
  const byNumber = new Map(segments.map((segment) => [Number(segment.segment), segment]));
  const windows = [];
  for (const occurrence of occurrences) {
    const segment = byNumber.get(Number(occurrence.stepSegment));
    const window = segment ? segmentWindow(segment, clock) : null;
    if (!window) continue;
    windows.push({ occurrence, ...window });
  }
  if (!windows.length) return 'no segment could be placed in the recording';
  let results;
  try {
    results = await analyseVideoWindows(video, windows, { crops, reference, stepSeconds, maxWidth });
  } catch (err) {
    // An unreadable recording is said on every segment it was meant to check, never left looking clean.
    const reason = safeMessage(err);
    for (const { occurrence } of windows) {
      occurrence.picture = { verdict: 'error', frozen: null, worstFrame: null, summary: `picture: error — ${reason}` };
    }
    return `the recording could not be read (${reason}); 0 segments checked`;
  }
  let checked = 0;
  const notChecked = {};
  for (let index = 0; index < windows.length; index += 1) {
    const { occurrence } = windows[index];
    const result = results[index];
    occurrence.picture = {
      verdict: result.verdict,
      frozen: result.frozen,
      worstFrame: result.worstFrame,
      summary: result.summary,
      ...(result.incomplete ? { incomplete: result.incomplete } : {}),
    };
    if (FINDINGS.includes(result.verdict)) occurrence.rejected = ['picture'];
    if (COULD_NOT_CHECK.includes(result.verdict)) notChecked[result.verdict] = (notChecked[result.verdict] ?? 0) + 1;
    else checked += 1;
  }
  const missed = Object.entries(notChecked).map(([verdict, count]) => `${count} ${verdict}`);
  return `${checked} segment${checked === 1 ? '' : 's'} checked${missed.length ? `; not checked: ${missed.join(', ')}` : ''}`;
}

// ----------------------------------------------------------------------------------------------------------------
// command line

// One flag per threshold (DEFAULTS explains each): [key, flag, min, max].
const THRESHOLD_FLAGS = [
  ['blackThreshold', '--black-threshold', 0, 255],
  ['blackAmount', '--black-amount', 0, 100],
  ['blackRise', '--black-rise', 0, 100],
  ['ssimMin', '--ssim-min', 0, 1],
  ['referenceMaxAgeSeconds', '--reference-max-age', 0, null],
  ['freezeNoise', '--freeze-noise', 0, 1],
  ['freezeSeconds', '--freeze-seconds', 0, null],
  ['decimateHi', '--decimate-hi', 0, null],
  ['decimateLo', '--decimate-lo', 0, null],
  ['decimateFrac', '--decimate-frac', 0, 1],
];

function usage(stream = process.stderr) {
  stream.write(
    'usage: node picture-check.mjs check <frame.png | recording.mp4> [--crops <json>] [--reference <image>]\n' +
      '       [--reference-at <s> (recording) | --reference-age <s> (image)] [--from <s>] [--to <s>] [--step <s>]\n' +
      '       [--max-width <px>] [--json] [threshold flags]\n' +
      `threshold flags (the header explains each): ${THRESHOLD_FLAGS.map(([, flag]) => flag).join(' ')}\n` +
      '       node picture-check.mjs reference --frame <image> [--out <png>]\n' +
      '       node picture-check.mjs reference --video <recording.mp4> --at <s> [--out <png>]\n' +
      'check exits 0 clean, 1 findings, 2 bad arguments, 3 could not check (unchecked, unlike the reference frame,\n' +
      'no frames, incomplete, unreadable input); reference exits 0 written, 2 bad arguments, 3 could not read the frame.\n',
  );
}

function pathDefaults(env, crops, reference) {
  const directory = path.join(capturesRoot(env), 'reference');
  const cropsFile = crops ?? path.join(directory, 'crops.json');
  return { cropsFile, referenceFile: reference ?? path.join(path.dirname(cropsFile), 'reference.png') };
}

const IMAGE_RE = /\.(?:png|bmp|jpe?g|webp|tif?f)$/i;
const VIDEO_RE = /\.(?:mp4|mov|mkv|webm|avi|m4v)$/i;

function numberOption(args, name, min, max) {
  const at = args.indexOf(name);
  if (at < 0) return null;
  const value = args[at + 1];
  args.splice(at, 2);
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < min || (max !== null && parsed > max)) {
    throw Object.assign(new Error(`${name} needs a number from ${min} to ${max ?? 'no limit'}`), { exitCode: 2 });
  }
  return parsed;
}

function textOption(args, name) {
  const at = args.indexOf(name);
  if (at < 0) return null;
  if (at + 1 >= args.length) throw Object.assign(new Error(`${name} needs a value`), { exitCode: 2 });
  const value = args[at + 1];
  args.splice(at, 2);
  return value;
}

export async function main(argv = process.argv.slice(2), env = process.env) {
  const args = [...argv];
  const asJson = args.includes('--json');
  const jsonAt = args.indexOf('--json');
  if (jsonAt >= 0) args.splice(jsonAt, 1);
  const command = args.shift();
  if (!command || command === '--help' || command === '-h') {
    usage(command ? process.stdout : process.stderr);
    return command ? 0 : 2;
  }
  if (command !== 'check' && command !== 'reference') {
    console.error(`picture-check: unknown command '${command}'`);
    usage();
    return 2;
  }
  try {
    loadLocalEnv({ env });
    const cropsOption = textOption(args, '--crops');
    const referenceOption = textOption(args, '--reference');
    const outOption = textOption(args, '--out');
    const stepSeconds = numberOption(args, '--step', 0.01, 3600) ?? DEFAULTS.stepSeconds;
    const fromSeconds = numberOption(args, '--from', 0, null) ?? 0;
    const toOption = numberOption(args, '--to', 0, null);
    const atSeconds = numberOption(args, '--at', 0, null);
    const maxWidth = numberOption(args, '--max-width', 8, null) ?? DEFAULTS.maxWidth;
    const referenceAt = numberOption(args, '--reference-at', 0, null);
    const referenceAge = numberOption(args, '--reference-age', 0, null);
    const options = { stepSeconds, fromSeconds, toSeconds: toOption, maxWidth };
    for (const [key, flag, min, max] of THRESHOLD_FLAGS) options[key] = numberOption(args, flag, min, max) ?? DEFAULTS[key];
    const defaults = pathDefaults(env, cropsOption, referenceOption);

    if (command === 'reference') {
      const frameOption = textOption(args, '--frame');
      const videoOption = textOption(args, '--video');
      if (frameOption && videoOption) throw Object.assign(new Error('reference takes --frame or --video, not both'), { exitCode: 2 });
      if (!frameOption && !videoOption) {
        throw Object.assign(new Error('reference needs --frame <image> or --video <recording.mp4> --at <s>'), { exitCode: 2 });
      }
      if (videoOption && atSeconds === null) throw Object.assign(new Error('reference --video needs --at <seconds>'), { exitCode: 2 });
      const size = writeReference({ frame: frameOption, video: videoOption, atSeconds: atSeconds ?? 0, out: outOption ?? defaults.referenceFile });
      console.log(`picture reference: a ${size.width}x${size.height} frame written`);
      return 0;
    }

    if (args.length !== 1) throw Object.assign(new Error('check needs one image or recording'), { exitCode: 2 });
    const input = args[0];
    const crops = loadCrops(defaults.cropsFile);
    const reference = loadReference(defaults.referenceFile);
    const bad = (message) => Object.assign(new Error(message), { exitCode: 2 });
    let result;
    if (IMAGE_RE.test(input)) {
      if (referenceAt !== null) throw bad('--reference-at is a time in a recording; an image takes --reference-age');
      result = await analyseImage(input, { ...options, crops, reference, referenceAgeSeconds: referenceAge });
    } else if (VIDEO_RE.test(input)) {
      if (referenceAge !== null) throw bad('--reference-age is for an image; a recording takes --reference-at');
      result = await analyseVideo(input, { ...options, crops, reference, referenceAtSeconds: referenceAt });
    } else {
      throw bad('check needs a PNG/image or a recording (mp4, mov, mkv, webm, avi)');
    }
    if (asJson) {
      console.log(JSON.stringify(result, null, 2));
    } else {
      console.log(result.summary);
      const filters = result.filters;
      console.log(
        `filters: ${filters.blackframe} (black from ${options.blackAmount}%, or ${points(options.blackRise)} over the ` +
          `reference crop's share), ${filters.freezedetect} and ${filters.mpdecimate} (frozen when both hold ${options.freezeSeconds} s), ` +
          `${filters.ssim} (differs under ${options.ssimMin} against a reference within ${options.referenceMaxAgeSeconds} s; ` +
          `older or of unknown age: ${UNLIKE}), ${filters.psnr}`,
      );
    }
    return exitCodeFor(result.verdict);
  } catch (err) {
    // The message never carries a path: the tool's output may leave this PC. Anything that stopped the check
    // (an unreadable input among it) is could-not-check, never a pass.
    console.error(`picture-check: ${safeMessage(err)}`);
    return err.exitCode || 3;
  }
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  main().then(
    (code) => {
      process.exitCode = code;
    },
    (err) => {
      console.error(`picture-check: ${safeMessage(err)}`);
      process.exitCode = 3;
    },
  );
}
