import assert from 'node:assert/strict';
import test from 'node:test';
import { TOOLS, check, expandFolder, lookUp, tidy } from './tools-check.mjs';

const HOME = 'C:/Users/someone';

// A fake machine: a set of files, a folder listing built from them, and a process runner that records its calls.
function machine(files, { env = {}, runs = {} } = {}) {
  const set = new Set(files.map((f) => f.toLowerCase()));
  const calls = [];
  return {
    calls,
    probe: {
      env,
      home: HOME,
      delimiter: ';',
      exists: (file) => set.has(file.toLowerCase()),
      readdir: (dir) => {
        const prefix = `${dir.toLowerCase()}/`;
        const names = new Set();
        for (const file of files) {
          if (file.toLowerCase().startsWith(prefix)) names.add(file.slice(prefix.length).split('/')[0]);
        }
        if (!names.size) throw new Error('ENOENT');
        return [...names];
      },
      run: (file, args, extra) => {
        calls.push({ file, args, extra });
        return runs[file] ?? { status: 1, stdout: '' };
      },
    },
  };
}

const tool = (name) => TOOLS.find((t) => t.name === name);

test('the required set ready: PresentMon by its variable, ffmpeg and ffprobe on PATH', () => {
  const pm = `${HOME}/tools/PresentMon-2.6.0-x64.exe`;
  const { probe, calls } = machine([pm, 'D:/bin/ffmpeg.exe', 'D:/bin/ffprobe.exe'], {
    env: { PRESENTMON_EXE: pm, PATH: 'C:/Windows;D:/bin' },
    runs: { 'D:/bin/ffmpeg.exe': { status: 0, stdout: 'ffmpeg version 9.1-full_build-www.gyan.dev Copyright\n' } },
  });
  const { lines, ok } = check(probe);
  assert.equal(ok, true);
  assert.equal(lines[0], 'PresentMon (required): found, 2.6.0, ~/tools/PresentMon-2.6.0-x64.exe (PRESENTMON_EXE)');
  assert.equal(lines[1], 'ffmpeg (required): found, 9.1-full_build-www.gyan.dev, D:/bin/ffmpeg.exe (PATH)');
  assert.equal(lines.length, TOOLS.length, 'one line per tool');
  assert.ok(!lines.join('\n').includes(HOME), 'the home folder is never printed');
  assert.deepEqual(calls.map((c) => c.args), [['-version']], 'the version came from the file name for PresentMon');
});

test('a missing required tool fails the check and says where it looked', () => {
  const { probe } = machine([], { env: { PATH: 'C:/Windows' } });
  const { lines, ok } = check(probe);
  assert.equal(ok, false);
  assert.equal(lines[0], 'PresentMon (required): missing; looked in PRESENTMON_EXE, PATH, ~/AppData/Local/Microsoft/WinGet/Links');
  assert.match(lines[1], /^ffmpeg \(required\): missing; looked in PATH, ~\/AppData\/Local\/Microsoft\/WinGet\/Links$/);
});

test('PresentMon found only on PATH is not used by the harness: the check fails and names the variable', () => {
  const { probe } = machine(['D:/bin/PresentMon.exe', 'D:/bin/ffmpeg.exe', 'D:/bin/ffprobe.exe'], { env: { PATH: 'D:/bin' } });
  const { lines, ok } = check(probe);
  assert.equal(ok, false);
  assert.match(lines[0], /found, version unknown, D:\/bin\/PresentMon\.exe \(PATH\); set PRESENTMON_EXE in harness\/local\.env/);
});

test('ffmpeg in the winget folder but not on PATH, or without ffprobe beside it, is not ready', () => {
  const links = `${HOME}/AppData/Local/Microsoft/WinGet/Links`;
  const pm = 'D:/pm/PresentMon-2.6.0-x64.exe';
  const offPath = machine([pm, `${links}/ffmpeg.exe`, `${links}/ffprobe.exe`], { env: { PRESENTMON_EXE: pm, PATH: '' } });
  const first = check(offPath.probe);
  assert.equal(first.ok, false);
  assert.match(first.lines[1], /\(the default folder\); put its folder on PATH/);
  const noProbe = machine([pm, 'D:/bin/ffmpeg.exe'], { env: { PRESENTMON_EXE: pm, PATH: 'D:/bin' } });
  const second = check(noProbe.probe);
  assert.equal(second.ok, false);
  assert.match(second.lines[1], /ffprobe\.exe not beside it/);
});

test('an unset or wrong variable falls through to PATH and then the default folder', () => {
  const ngfx = 'C:/Program Files/NVIDIA Corporation/Nsight Graphics 2026.4.0/host/windows-desktop-nomad-x64/ngfx.exe';
  const older = 'C:/Program Files/NVIDIA Corporation/Nsight Graphics 2025.2.1/host/windows-desktop-nomad-x64/ngfx.exe';
  const { probe } = machine([older, ngfx], { env: { VICTORY_NGFX: 'D:/nowhere/ngfx.exe', PATH: '' } });
  const result = lookUp(tool('Nsight Graphics'), probe);
  assert.equal(result.found, ngfx, 'the newest version folder');
  assert.equal(result.via, 'the default folder');
  assert.equal(result.usable, true, 'gpu-trace.mjs finds the default folder itself');
  assert.deepEqual(result.looked, ['VICTORY_NGFX', 'PATH', 'C:/Program Files/NVIDIA Corporation/Nsight Graphics */host/windows-desktop-nomad-x64']);
  assert.match(check(probe, [tool('Nsight Graphics')]).lines[0], /^Nsight Graphics: found, 2026\.4\.0, /);
});

test('RENDERDOC_DIR names a folder; PIX is versioned by its folder; a file version is read through the environment', () => {
  const { probe, calls } = machine(['E:/rd/RenderDoc_1.47_64/renderdoccmd.exe', 'C:/Program Files/Microsoft PIX/2611.19/pixtool.exe'], {
    env: { RENDERDOC_DIR: 'E:/rd/RenderDoc_1.47_64/', PATH: '' },
    runs: { powershell: { status: 0, stdout: '1.47\r\n' } },
  });
  const { lines } = check(probe, [tool('RenderDoc'), tool('PIX on Windows')]);
  assert.equal(lines[0], 'RenderDoc: found, 1.47, E:/rd/RenderDoc_1.47_64/renderdoccmd.exe (RENDERDOC_DIR)');
  assert.equal(lines[1], 'PIX on Windows: found, 2611.19, C:/Program Files/Microsoft PIX/2611.19/pixtool.exe (the default folder)');
  assert.equal(calls.length, 1);
  assert.equal(calls[0].file, 'powershell');
  assert.ok(!calls[0].args.join(' ').includes('E:/rd'), 'the path is not on the command line');
  assert.equal(calls[0].extra.TOOLS_CHECK_FILE, 'E:/rd/RenderDoc_1.47_64/renderdoccmd.exe');
});

test('the Windows Performance Toolkit notes xperf when it is not beside the analyzer; optional tools never fail the check', () => {
  const pm = 'D:/pm/PresentMon-2.6.0-x64.exe';
  const { probe } = machine([pm, 'D:/bin/ffmpeg.exe', 'D:/bin/ffprobe.exe', 'D:/store/wpa.exe'], {
    env: { PRESENTMON_EXE: pm, PATH: 'D:/bin;D:/store' },
  });
  const { lines, ok } = check(probe);
  assert.equal(ok, true);
  assert.match(lines[2], /^Windows Performance Toolkit: found, version unknown, D:\/store\/wpa\.exe \(PATH\); xperf\.exe not beside it$/);
  assert.match(lines[3], /^Tracy: missing; looked in PATH$/);
});

test('tidy replaces the home folder in either slash style and any case, and leaves other paths alone', () => {
  assert.equal(tidy('C:\\Users\\someone\\tools\\x.exe', HOME), '~/tools/x.exe');
  assert.equal(tidy('c:/users/SOMEONE/x', HOME), '~/x');
  assert.equal(tidy('C:/Users/someoneelse/x', HOME), 'C:/Users/someoneelse/x');
  assert.equal(tidy('D:/bin/ffmpeg.exe', HOME), 'D:/bin/ffmpeg.exe');
});

test('a folder pattern expands newest first, and a missing parent is no match', () => {
  const { probe } = machine(['C:/A/v 1.9/x', 'C:/A/v 1.10/x', 'C:/A/other/x']);
  assert.deepEqual(expandFolder('C:/A/v */x', probe), ['C:/A/v 1.10/x', 'C:/A/v 1.9/x']);
  assert.deepEqual(expandFolder('C:/B/*', probe), []);
  assert.deepEqual(expandFolder('~/AppData', probe), [`${HOME}/AppData`]);
});
