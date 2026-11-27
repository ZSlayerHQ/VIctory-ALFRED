import assert from 'node:assert/strict';
import test from 'node:test';
import { ffmpegArgs, startedLine } from './record-screen.mjs';

test('the ffmpeg argument list is the proven gfxcapture command plus the H.264 options', () => {
  assert.deepEqual(ffmpegArgs(123456, 'D:/captures/run/screen.mp4'), [
    '-hide_banner',
    '-v',
    'error',
    '-y',
    '-f',
    'lavfi',
    '-i',
    'gfxcapture=hwnd=123456:max_framerate=2:capture_cursor=0',
    '-vf',
    'hwdownload,format=bgra,scale=1920:-2',
    '-r',
    '2',
    '-c:v',
    'libx264',
    '-crf',
    '23',
    '-pix_fmt',
    'yuv420p',
    '-movflags',
    '+frag_keyframe+empty_moov+default_base_moof',
    '-g',
    '4',
    'D:/captures/run/screen.mp4',
  ]);
});

test('no handle, or no gfxcapture filter, means do not record', () => {
  assert.equal(ffmpegArgs(0, 'screen.mp4'), null);
  assert.equal(ffmpegArgs(null, 'screen.mp4'), null);
  assert.equal(ffmpegArgs(123456, 'screen.mp4', { hasGfxcapture: false }), null);
});

test('the run log names the wall-clock time the recording began', () => {
  assert.equal(
    startedLine(Date.parse('2026-11-25T02:53:29.602Z')),
    'recording: started 2026-11-25T02:53:29.602Z',
  );
});
