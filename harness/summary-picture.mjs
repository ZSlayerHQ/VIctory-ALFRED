// The run summary's picture block, from the picture result summarize-run.mjs builds out of picture-check.mjs.
// Kept apart from summarize-run.mjs (a script that runs on import) so a test can hand it a made-up result.
import { clock } from './screen-timeline.mjs';

// A field absent from the result prints n/a: never zero, never a dash that reads as a blank.
const NA = 'n/a';
const f = (x, d = 1) => (x === null || x === undefined || !Number.isFinite(Number(x)) ? NA : Number(x).toFixed(d));

// picture-check's per-crop figures (VERSION 3, ffmpeg's blackframe, ssim and psnr): blackShare is 0..1, psnr is in
// dB and infinite for identical pictures.
const black = (share) => (share === null || share === undefined || !Number.isFinite(share) ? NA : `${(share * 100).toFixed(1)}%`);
const psnr = (db) => (db === Infinity ? 'inf dB' : db === null || db === undefined || !Number.isFinite(db) ? NA : `${db.toFixed(1)} dB`);
const cropFigures = (crop) => `black ${black(crop.blackShare)}, ssim ${f(crop.ssim, 3)}, psnr ${psnr(crop.psnr)}`;

/// The block's lines, heading included; none when there is no picture result.
export function pictureLines(picture) {
  if (!picture) return [];
  const out = ['\n## Picture'];
  if (picture.verdict === 'no-reference' || picture.verdict === 'unavailable') {
    out.push(`- ${picture.summary}`);
    return out;
  }
  const frozen = picture.frozen && picture.frozen.verdict === 'frozen' ? `, frozen ${f(picture.frozen.seconds)} s` : '';
  out.push(`- verdict: ${picture.verdict} — ${picture.frames} frames${frozen}`);
  for (const [name, crop] of Object.entries(picture.crops ?? {})) out.push(`- ${name}: ${crop.verdict} — ${cropFigures(crop)}`);
  if (picture.worstFrame) {
    const worst = picture.worstFrame;
    const parts = Object.entries(worst.crops ?? {})
      .map(([name, crop]) => `${name} ${cropFigures(crop)}`)
      .join('; ');
    out.push(`- worst frame ${worst.index} at ${clock(worst.timeSeconds)}: ${worst.verdict} — ${parts}`);
  }
  return out;
}
