#!/usr/bin/env node
// Which tools of the diagnostic pack (docs/DIAGNOSTICS.md) this machine has, and where they were found.
//
//   node harness/tools-check.mjs
//
// One line per tool: found or missing, its version, and where it was looked for: the harness's own variable (from
// the environment or harness/local.env), then PATH, then the tool's default install folder. Exit 0 when the harness
// can use the required set (PresentMon through PRESENTMON_EXE; ffmpeg and ffprobe on PATH), 1 otherwise. A path in
// the home folder is printed with `~` in its place. `check` takes every lookup as a function, so the tests start no
// process and read no real folder.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { loadLocalEnv } from './local-env.mjs';

// `use` is how the harness finds the tool: `env` reads only the variable, `path` calls it by name from PATH, `any`
// takes the variable and else the default folder (gpu-trace.mjs), and null means the harness does not start it.
// A folder may hold `*` in a segment; the newest match by version order is taken. `~` is the home folder.
export const TOOLS = [
  // winget's portable packages (PresentMon's console app, ffmpeg) leave a link in WinGet/Links, which winget puts on PATH.
  {
    name: 'PresentMon',
    required: true,
    env: 'PRESENTMON_EXE',
    envKind: 'file',
    use: 'env',
    files: ['PresentMon.exe'],
    folders: ['~/AppData/Local/Microsoft/WinGet/Links'],
  },
  {
    name: 'ffmpeg',
    required: true,
    use: 'path',
    files: ['ffmpeg.exe'],
    also: ['ffprobe.exe'],
    folders: ['~/AppData/Local/Microsoft/WinGet/Links'],
    version: { args: ['-version'], pattern: /^ffmpeg version (\S+)/m },
  },
  {
    name: 'Windows Performance Toolkit',
    use: null,
    files: ['wpa.exe'],
    also: ['xperf.exe'],
    folders: ['C:/Program Files (x86)/Windows Kits/10/Windows Performance Toolkit'],
  },
  { name: 'Tracy', use: null, files: ['tracy-profiler.exe'], folders: [] },
  // The MSI's usual folder; RenderDoc's page does not state it, so the portable zip is found through RENDERDOC_DIR.
  { name: 'RenderDoc', env: 'RENDERDOC_DIR', envKind: 'dir', use: 'env', files: ['renderdoccmd.exe'], folders: ['C:/Program Files/RenderDoc'] },
  { name: 'PIX on Windows', use: null, files: ['pixtool.exe'], folders: ['C:/Program Files/Microsoft PIX/*'], versionFrom: /Microsoft PIX\/([^/]+)/ },
  {
    name: 'Nsight Graphics',
    env: 'VICTORY_NGFX',
    envKind: 'file',
    use: 'any',
    files: ['ngfx.exe'],
    folders: ['C:/Program Files/NVIDIA Corporation/Nsight Graphics */host/windows-desktop-nomad-x64'],
    versionFrom: /Nsight Graphics ([^/]+)/,
  },
];

const slash = (p) => String(p).split('\\').join('/');

/// `p` with the home folder replaced by `~`, in forward slashes.
export function tidy(p, home) {
  const text = slash(p);
  const root = slash(home ?? '').replace(/\/+$/, '');
  if (root && (text.toLowerCase() === root.toLowerCase() || text.toLowerCase().startsWith(`${root.toLowerCase()}/`))) {
    return `~${text.slice(root.length)}`;
  }
  return text;
}

const versionOrder = (a, b) => b.localeCompare(a, undefined, { numeric: true });

/// The folders a pattern names, newest first: each `*` segment is matched against the folder's entries.
export function expandFolder(pattern, { home, readdir }) {
  let found = [pattern.startsWith('~') ? `${slash(home)}${pattern.slice(1)}` : pattern];
  if (!found[0].includes('*')) return found;
  const segments = found[0].split('/');
  found = [segments[0]];
  for (const segment of segments.slice(1)) {
    if (!segment.includes('*')) {
      found = found.map((dir) => `${dir}/${segment}`);
      continue;
    }
    const match = new RegExp(`^${segment.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*')}$`);
    found = found.flatMap((dir) => {
      let names;
      try {
        names = readdir(dir);
      } catch {
        return [];
      }
      return names.filter((name) => match.test(name)).sort(versionOrder).map((name) => `${dir}/${name}`);
    });
  }
  return found;
}

/// One tool's lookup. Returns { name, required, found, usable, where, via, version, missingAlso, looked }.
export function lookUp(tool, probe) {
  const { env, exists, readdir, home } = probe;
  const looked = [];
  let found = null;
  let via = null;
  if (tool.env) {
    looked.push(tool.env);
    const value = env[tool.env];
    if (value) {
      const file = tool.envKind === 'dir' ? `${slash(value).replace(/\/+$/, '')}/${tool.files[0]}` : slash(value);
      if (exists(file)) [found, via] = [file, tool.env];
    }
  }
  if (!found) {
    looked.push('PATH');
    const dirs = String(env.PATH ?? env.Path ?? '').split(probe.delimiter ?? path.delimiter).filter(Boolean);
    outer: for (const dir of dirs) {
      for (const name of tool.files) {
        const file = `${slash(dir).replace(/\/+$/, '')}/${name}`;
        if (exists(file)) {
          [found, via] = [file, 'PATH'];
          break outer;
        }
      }
    }
  }
  if (!found) {
    for (const pattern of tool.folders) {
      looked.push(pattern);
      for (const dir of expandFolder(pattern, { home, readdir })) {
        const file = tool.files.map((name) => `${dir}/${name}`).find(exists);
        if (file) {
          [found, via] = [file, 'the default folder'];
          break;
        }
      }
      if (found) break;
    }
  }
  const usable = !!found && (tool.use === 'env' ? via === tool.env : tool.use === 'path' ? via === 'PATH' : tool.use === 'any' ? via !== 'PATH' : true);
  const missingAlso = found
    ? (tool.also ?? []).filter((name) => !exists(`${path.posix.dirname(found)}/${name}`))
    : [];
  return { name: tool.name, required: !!tool.required, found, usable, via, missingAlso, looked };
}

/// A found tool's version, or null: from the folder name, the file name, the program's own output, or the file's
/// version resource (PowerShell reads it; the path goes in through the environment, never on a command line).
export function versionOf(tool, file, { run }) {
  if (tool.versionFrom) return tool.versionFrom.exec(slash(file))?.[1] ?? null;
  const named = /(\d+\.\d+(?:\.\d+)*)/.exec(path.posix.basename(slash(file)));
  if (named) return named[1];
  if (tool.version) {
    const result = run(file, tool.version.args, {});
    return result.status === 0 ? (tool.version.pattern.exec(result.stdout ?? '')?.[1] ?? null) : null;
  }
  const result = run(
    'powershell',
    ['-NoProfile', '-NonInteractive', '-Command', '(Get-Item -LiteralPath $env:TOOLS_CHECK_FILE).VersionInfo.ProductVersion'],
    { TOOLS_CHECK_FILE: file },
  );
  const text = result.status === 0 ? String(result.stdout ?? '').trim() : '';
  return text || null;
}

/// The printed line for one result.
export function lineFor(tool, result, version, home) {
  const label = `${tool.name}${tool.required ? ' (required)' : ''}`;
  if (!result.found) return `${label}: missing; looked in ${result.looked.join(', ')}`;
  let line = `${label}: found, ${version ?? 'version unknown'}, ${tidy(result.found, home)} (${result.via})`;
  if (result.missingAlso.length) line += `; ${result.missingAlso.join(', ')} not beside it`;
  if (!result.usable) {
    line +=
      tool.use === 'path'
        ? '; put its folder on PATH, the harness calls it by name'
        : `; set ${tool.env} in harness/local.env for the harness to use it`;
  }
  return line;
}

/// Every tool's line and whether the required set is usable. `probe`: { env, exists, readdir, run, home, delimiter }.
export function check(probe, tools = TOOLS) {
  const lines = [];
  let ok = true;
  for (const tool of tools) {
    const result = lookUp(tool, probe);
    const version = result.found ? versionOf(tool, result.found, probe) : null;
    lines.push(lineFor(tool, result, version, probe.home));
    const complete = result.usable && (tool.use !== 'path' || result.missingAlso.length === 0);
    if (tool.required && !complete) ok = false;
  }
  return { lines, ok };
}

function realProbe(env) {
  return {
    env,
    home: os.homedir(),
    exists: (file) => {
      try {
        return fs.statSync(file).isFile();
      } catch {
        return false;
      }
    },
    readdir: (dir) => fs.readdirSync(dir),
    run: (file, args, extra) => {
      const result = spawnSync(file, args, { encoding: 'utf8', timeout: 10000, windowsHide: true, env: { ...env, ...extra } });
      return { status: result.status, stdout: result.stdout };
    },
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  loadLocalEnv();
  const { lines, ok } = check(realProbe(process.env));
  for (const line of lines) console.log(line);
  console.log(ok ? 'tools-check: the required set is ready' : 'tools-check: the required set is not ready (docs/DIAGNOSTICS.md)');
  process.exit(ok ? 0 : 1);
}
