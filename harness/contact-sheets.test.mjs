import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'contact-sheets.mjs');

test('a three-second video gives one non-empty sheet of its one-a-second tiles', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'contact-sheets-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const video = path.join(dir, 'screen.mp4');
  const made = spawnSync(
    'ffmpeg',
    ['-hide_banner', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=size=320x180:rate=2:duration=3', '-pix_fmt', 'yuv420p', video],
    { encoding: 'utf8' },
  );
  assert.equal(made.status, 0, made.stderr);

  const result = spawnSync(process.execPath, [script, video], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /contact sheets: 1 in /);

  const sheets = fs
    .readdirSync(path.join(dir, 'sheets'))
    .filter((file) => /^sheet-\d+\.jpg$/.test(file));
  assert.deepEqual(sheets, ['sheet-001.jpg']);
  assert.ok(fs.statSync(path.join(dir, 'sheets', 'sheet-001.jpg')).size > 0);
});
