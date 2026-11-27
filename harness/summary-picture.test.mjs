// The summary's picture block on made-up picture results (no recording is decoded).
import assert from 'node:assert/strict';
import test from 'node:test';

import { pictureLines } from './summary-picture.mjs';

// The shape summarize-run.mjs builds from picture-check.mjs (VERSION 3, the ffmpeg filters): per crop a verdict
// with blackShare (0..1), ssim and psnr against the reference.
const result = {
  verdict: 'differs',
  reference: true,
  frames: 30,
  frozen: { verdict: 'clean', seconds: 0 },
  crops: {
    flag: { verdict: 'differs', blackShare: 0.125, ssim: 0.6123, psnr: 18.4 },
    sky: { verdict: 'clean', blackShare: 0, ssim: 0.9876, psnr: 31.25 },
    dock: { verdict: 'unchecked', blackShare: null, ssim: null, psnr: null },
  },
  worstFrame: {
    index: 12,
    timeSeconds: 24,
    verdict: 'differs',
    crops: {
      flag: { verdict: 'differs', blackShare: 0.125, ssim: 0.6123, psnr: 18.4 },
      sky: { verdict: 'clean', blackShare: 0, ssim: 0.9876, psnr: 31.25 },
    },
  },
};

test('each crop prints its black share, SSIM and PSNR, and n/a where the result has none', () => {
  const lines = pictureLines(result);
  assert.ok(lines.includes('- flag: differs — black 12.5%, ssim 0.612, psnr 18.4 dB'), lines.join('\n'));
  assert.ok(lines.includes('- sky: clean — black 0.0%, ssim 0.988, psnr 31.3 dB'));
  assert.ok(lines.includes('- dock: unchecked — black n/a, ssim n/a, psnr n/a'));
  assert.match(lines.find((line) => line.startsWith('- worst frame')), /^- worst frame 12 at 0:24: differs — flag black 12\.5%, ssim 0\.612, psnr 18\.4 dB; sky black 0\.0%, ssim 0\.988, psnr 31\.3 dB$/);
  for (const line of lines) assert.doesNotMatch(line, /undefined|NaN|near-black/, line);
});

test('no reference and unavailable keep their one line', () => {
  assert.deepEqual(pictureLines({ verdict: 'no-reference', summary: 'picture: no reference' }), ['\n## Picture', '- picture: no reference']);
  assert.deepEqual(pictureLines(null), []);
});
