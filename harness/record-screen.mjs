#!/usr/bin/env node
// Records this run's game window at two frames a second into screen.mp4, so every run leaves a record of what was
// on screen (the maintainer, 2026-11-25; G-017). The window is selected the run's way, run-process.ps1 -Strict: a victory.exe
// created after this run began, with " run " and the game's path on its command line, and no fallback. It never
// records a monitor or another window, and it ends only its own ffmpeg child (G-016).
//
// usage: node record-screen.mjs <output.mp4>
// Needs RUN_WORLD_GAME_EXE and RUN_WORLD_STARTED_EPOCH in the environment (run-world.sh sets both).
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const FFMPEG = 'ffmpeg';

// The run log's marker for when the recording began: screen-timeline.mjs joins the log to the video by it.
export const startedLine = (at) => `recording: started ${new Date(at).toISOString()}`;

// The ffmpeg argument list: the lead's proven gfxcapture command plus the H.264 output options. `null` means do
// not record (no window handle, or no gfxcapture filter).
export function ffmpegArgs(handle, output, { hasGfxcapture = true } = {}) {
  if (!hasGfxcapture || !handle) return null;
  return [
    '-hide_banner',
    '-v',
    'error',
    '-y',
    '-f',
    'lavfi',
    '-i',
    `gfxcapture=hwnd=${handle}:max_framerate=2:capture_cursor=0`,
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
    // Fragmented, so the file can be opened while the run is still going and survives a run that is ended hard.
    '-movflags',
    '+frag_keyframe+empty_moov+default_base_moof',
    '-g',
    '4',
    output,
  ];
}

export function hasGfxcapture(ffmpeg = FFMPEG) {
  const r = spawnSync(ffmpeg, ['-hide_banner', '-filters'], { encoding: 'utf8' });
  return r.status === 0 && /gfxcapture/.test(r.stdout || '');
}

// This run's game window, from the shared selection: "<pid> <handle>", or null. The helper waits up to 120 s.
function gameWindow() {
  const r = spawnSync(
    'powershell.exe',
    ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', path.join(here, 'window-handle.ps1')],
    { encoding: 'utf8', timeout: 130_000 },
  );
  if (r.status !== 0) return null;
  for (const line of (r.stdout || '').split(/\r?\n/)) {
    const m = line.match(/^(\d+)\s+(\d+)$/);
    if (m) return { pid: Number(m[1]), handle: Number(m[2]) };
  }
  return null;
}

function alive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function framesIn(file) {
  const r = spawnSync(
    'ffprobe',
    ['-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'default=nw=1:nk=1', file],
    { encoding: 'utf8' },
  );
  const n = Number((r.stdout || '').trim());
  return Number.isFinite(n) && n > 0 ? n : null;
}

function main() {
  const output = process.argv[2];
  if (!output) {
    console.error('usage: node record-screen.mjs <output.mp4>');
    process.exit(2);
  }
  if (!hasGfxcapture()) {
    console.log('recording: none (this ffmpeg has no gfxcapture filter)');
    return;
  }
  const win = gameWindow();
  if (!win) {
    console.log("recording: none (no window for this run's victory.exe within 120 s)");
    return;
  }
  const args = ffmpegArgs(win.handle, output);
  if (!args) {
    console.log('recording: none (no window handle)');
    return;
  }
  fs.mkdirSync(path.dirname(output), { recursive: true });
  const started = Date.now();
  const ff = spawn(FFMPEG, args, { stdio: ['pipe', 'ignore', 'pipe'] });
  let stderr = '';
  ff.stderr.on('data', (d) => {
    stderr += d;
  });
  console.log(startedLine(started));
  console.log(`recording: ${output}, window of pid ${win.pid}`);

  let finishing = false;
  const finish = () => {
    if (finishing) return;
    finishing = true;
    clearInterval(poll);
    const seconds = ((Date.now() - started) / 1000).toFixed(1);
    try {
      ff.stdin.write('q\n');
    } catch {
      // ffmpeg already ended; the exit handler below reports it.
    }
    // Only this child: a clean quit first, then end it if it will not go.
    const timer = setTimeout(() => {
      try {
        ff.kill();
      } catch {
        // already ended
      }
    }, 10_000);
    ff.on('exit', () => {
      clearTimeout(timer);
      const frames = framesIn(output);
      const detail = stderr.trim() ? ` (ffmpeg: ${stderr.trim().split('\n')[0]})` : '';
      console.log(
        `recording ended: ${frames === null ? '~' + Math.round(Number(seconds) * 2) : frames} frames, ${seconds} s${detail}`,
      );
      process.exit(0);
    });
  };
  const poll = setInterval(() => {
    if (!alive(win.pid)) finish();
  }, 500);
  ff.on('exit', () => {
    if (!finishing) {
      finishing = true;
      clearInterval(poll);
      const detail = stderr.trim() ? stderr.trim().split('\n')[0] : 'no error';
      console.log(`recording: stopped by ffmpeg (${detail})`);
      process.exit(0);
    }
  });
  process.on('SIGTERM', finish);
  process.on('SIGINT', finish);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
