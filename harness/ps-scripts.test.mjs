import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

function powershell(script) {
  return spawnSync('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', script], {
    encoding: 'utf8',
  });
}

test('every harness PowerShell script parses', () => {
  const files = fs.readdirSync(here).filter((file) => file.endsWith('.ps1'));
  assert.ok(files.length > 0);
  for (const name of files) {
    const file = path.join(here, name).replace(/'/g, "''");
    const script =
      `$errors = $null; [System.Management.Automation.Language.Parser]::ParseFile('${file}', [ref]$null, [ref]$errors) | Out-Null; ` +
      `if ($errors.Count -gt 0) { $errors | ForEach-Object { $_.Message }; exit 1 }`;
    const result = powershell(script);
    assert.equal(result.status, 0, `${name}: ${result.stdout}${result.stderr}`);
  }
});

// run-process.ps1's top-level param block resets a dot-sourcing caller's switches of the same names. That is the
// open fault; raise-priority.ps1 avoids it by dot-sourcing inside a child scope with the value captured first.
// These run PowerShell only: the dot-sourced path never reaches the process selection (G-016).
test('dot-sourcing run-process.ps1 resets a caller switch of the same name', () => {
  const runProcess = path.join(here, 'run-process.ps1').replace(/'/g, "''");
  const result = powershell(`$Translate = $true; . '${runProcess}'; if ($Translate) { 'after=on' } else { 'after=off' }`);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /after=off/);
});

// -Translate never falls back to another victory.exe, and without a warm-up start time it selects nothing, so the
// gentle switch can be run here without touching any process.
test('raise-priority.ps1 -Translate -Gentle with no warm-up started lowers nothing', () => {
  const file = path.join(here, 'raise-priority.ps1').replace(/'/g, "''");
  const result = powershell(`Remove-Item Env:RUN_WORLD_WARM_STARTED_EPOCH -ErrorAction SilentlyContinue; & '${file}' -Translate -Gentle`);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /^priority: no victory\.exe raised \(\d+ found\)/);
});

test('the child-scope wrapper keeps the caller switch', () => {
  const runProcess = path.join(here, 'run-process.ps1').replace(/'/g, "''");
  const result = powershell(
    `$Translate = $true; ` +
      `$inner = & { param($UseTranslate) . '${runProcess}'; if ($UseTranslate) { 'inner=on' } else { 'inner=off' } } $Translate; ` +
      `if ($Translate) { 'after=on' } else { 'after=off' }; $inner`,
  );
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /inner=on/);
  assert.match(result.stdout, /after=on/);
});

// set-volume.ps1 changes only the process it picked: a process id nobody holds changes nothing, and a stand-in with no
// audio of its own is left with nothing changed and the script ends when it does (it never outlives its process).
test('set-volume changes nothing for a process id that is gone, and stops when its process ends', { skip: process.platform !== 'win32' }, async () => {
  const { spawn } = await import('node:child_process');
  const script = path.join(here, 'set-volume.ps1');
  const child = spawn(process.execPath, ['-e', 'setTimeout(() => {}, 4000)'], { stdio: 'ignore' });
  const pid = child.pid;
  await new Promise((done) => child.on('exit', done));
  const gone = spawnSync('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script, '-Percent', '50', '-ProcessId', String(pid)], { encoding: 'utf8', timeout: 60000 });
  assert.equal(gone.status, 0, gone.stderr);
  assert.match(gone.stdout, new RegExp(`volume: pid ${pid} is gone; nothing changed`));
  const quiet = spawn(process.execPath, ['-e', 'setTimeout(() => {}, 5000)'], { stdio: 'ignore' });
  const started = Date.now();
  const watched = spawnSync('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script, '-Percent', '50', '-ProcessId', String(quiet.pid)], { encoding: 'utf8', timeout: 60000 });
  assert.equal(watched.status, 0, watched.stderr);
  assert.match(watched.stdout, /opened no audio session; nothing changed/);
  assert.ok(Date.now() - started < 30000, 'it ended soon after its process did');
});
