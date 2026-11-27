#!/usr/bin/env node
// Contact sheets from a run's screen.mp4: one tile a second, three by three by default, each stamped with its time
// from the start of the recording. These are what an agent or a person opens to review a whole run quickly.
// usage: node contact-sheets.mjs <screen.mp4> [--every 1] [--cols 3 --rows 3 --width 960]
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
function take(name, fallback) {
  const i = args.indexOf(name);
  if (i < 0) return fallback;
  const v = args[i + 1];
  args.splice(i, 2);
  return v === undefined ? fallback : v;
}
const every = Number(take('--every', '1'));
const cols = Number(take('--cols', '3'));
const rows = Number(take('--rows', '3'));
const width = Number(take('--width', '960'));
const video = args[0];
if (!video) {
  console.error('usage: node contact-sheets.mjs <screen.mp4> [--every 1] [--cols 3 --rows 3 --width 960]');
  process.exit(2);
}
if (!(every > 0) || !(cols > 0) || !(rows > 0) || !(width > 0)) {
  console.error('contact sheets: --every, --cols, --rows and --width take positive numbers');
  process.exit(2);
}

// drawtext needs a font file here: the ffmpeg build on this PC has no fontconfig default.
const fonts = ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/segoeui.ttf'];
const font = fonts.find((file) => fs.existsSync(file));
if (!font) {
  console.error('contact sheets: no font file found for the time captions');
  process.exit(1);
}
const fontFile = `fontfile='${font.replace(/:/g, '\\:')}'`;

// One tile a second; each sheet holds cols x rows tiles.
const tileWidth = Math.max(2, Math.round(width / cols));
const filter =
  `fps=${every},scale=${tileWidth}:-2,` +
  `drawtext=${fontFile}:text='%{pts\\:hms}':x=8:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.5,` +
  `tile=${cols}x${rows}`;

const dir = path.join(path.dirname(video), 'sheets');
fs.mkdirSync(dir, { recursive: true });
const result = spawnSync(
  'ffmpeg',
  ['-hide_banner', '-v', 'error', '-y', '-i', video, '-vf', filter, '-q:v', '3', path.join(dir, 'sheet-%03d.jpg')],
  { encoding: 'utf8' },
);
if (result.status !== 0) {
  const why = (result.stderr || '').trim().split('\n')[0] || 'no error';
  console.error(`contact sheets: ffmpeg failed: ${why}`);
  process.exit(1);
}
const sheets = fs.readdirSync(dir).filter((file) => /^sheet-\d+\.jpg$/.test(file)).length;
console.log(`contact sheets: ${sheets} in ${dir}`);
