import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { isolatedEnv } from './test-isolation.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(here, 'run-world.sh');

function tempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'run-world-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

// A harness run needs an executable and a folder outside every work tree; the shell itself stands in for the game
// binary, and --print-env stops the script before anything is started.
function printEnv(t, args, extraEnv = {}) {
  const dir = tempDir(t);
  const bash = spawnSync('bash', ['-c', 'command -v bash'], { encoding: 'utf8' }).stdout.trim();
  fs.writeFileSync(path.join(dir, 'victory.toml'), '');
  const result = spawnSync('bash', [script, ...args, '--print-env'], {
    encoding: 'utf8',
    env: isolatedEnv({
      VICTORY_BIN: bash,
      VICTORY_DIR: dir,
      VICTORY_CONFIG: path.join(dir, 'victory.toml'),
      GAME_EXE: path.join(dir, 'game.exe'),
      VICTORY_CAPTURES: path.join(dir, 'captures'),
      ...extraEnv,
    }),
  });
  assert.equal(result.status, 0, result.stderr);
  return new Set(result.stdout.split(/\r?\n/).map((line) => line.trim()).filter(Boolean));
}

function watchdogOf(env) {
  const line = [...env].find((entry) => entry.startsWith('watchdog='));
  return line ? line.slice('watchdog='.length) : null;
}

test('--dump-targets sets the unsafe flag beside the targets and keeps the frame list', (t) => {
  const env = printEnv(t, ['--dump-targets', '--frames', '2400,2600']);
  assert.equal(env.has('VICTORY_DUMP_TARGETS=1'), true);
  assert.equal(env.has('VICTORY_DUMP_UNSAFE=1'), true, 'the runtime refuses the targets without it');
  assert.equal(env.has('VICTORY_CAPTURE_FRAMES=2400,2600'), true);
});

test('without --dump-targets neither dump variable is passed', (t) => {
  const env = printEnv(t, ['--frames', '2400,2600']);
  for (const line of env) assert.doesNotMatch(line, /^VICTORY_DUMP_/);
});

test('a timed run defaults the watchdog to ending, an open session to warn', (t) => {
  assert.equal(watchdogOf(printEnv(t, [])), 'warn', 'an open, untimed session warns');
  assert.equal(watchdogOf(printEnv(t, ['--heavy-seconds', '60'])), '1', 'a heavy hold is timed');
  assert.equal(watchdogOf(printEnv(t, ['--settled-seconds', '60'])), '1', 'a settled hold is timed');
  // A route can supply the hold, and the default follows it. The route is a fixture in the test's own folder
  // (HARNESS_ROUTES_DIR), so the test needs no route a checkout or the public tree may lack.
  const routes = tempDir(t);
  fs.writeFileSync(path.join(routes, 'fixture-hold.keys'), '# settled-seconds: 30\n5:Enter\n');
  assert.equal(
    watchdogOf(printEnv(t, ['--route', 'fixture-hold'], { HARNESS_ROUTES_DIR: routes })),
    '1',
    'a route with a hold is timed',
  );
});

test('an explicit WATCHDOG=1, 0 or warn wins over either default', (t) => {
  for (const value of ['1', '0', 'warn']) {
    assert.equal(watchdogOf(printEnv(t, [], { WATCHDOG: value })), value, `open session, explicit ${value}`);
    assert.equal(
      watchdogOf(printEnv(t, ['--heavy-seconds', '60'], { WATCHDOG: value })),
      value,
      `timed run, explicit ${value}`,
    );
  }
});

test('a WATCHDOG value outside 1, 0 and warn is refused', (t) => {
  const dir = tempDir(t);
  fs.writeFileSync(path.join(dir, 'victory.toml'), '');
  const result = spawnSync('bash', [script, '--print-env'], {
    encoding: 'utf8',
    env: isolatedEnv({
      VICTORY_BIN: spawnSync('bash', ['-c', 'command -v bash'], { encoding: 'utf8' }).stdout.trim(),
      VICTORY_DIR: dir,
      VICTORY_CONFIG: path.join(dir, 'victory.toml'),
      GAME_EXE: path.join(dir, 'game.exe'),
      VICTORY_CAPTURES: path.join(dir, 'captures'),
      WATCHDOG: 'sometimes',
    }),
  });
  assert.equal(result.status, 2);
  assert.match(result.stderr, /refused: WATCHDOG is 1, 0 or warn, not 'sometimes'/);
});

test('a direct run is refused unless it holds the lock; --print-env and --help are allowed', (t) => {
  const dir = tempDir(t);
  fs.writeFileSync(path.join(dir, 'victory.toml'), '');
  const env = isolatedEnv({
    VICTORY_BIN: spawnSync('bash', ['-c', 'command -v bash'], { encoding: 'utf8' }).stdout.trim(),
    VICTORY_DIR: dir,
    VICTORY_CONFIG: path.join(dir, 'victory.toml'),
    GAME_EXE: path.join(dir, 'game.exe'),
    VICTORY_CAPTURES: path.join(dir, 'captures'),
  });
  const refused = spawnSync('bash', [script, '--tag', 'lockless', '--no-record'], { encoding: 'utf8', env });
  assert.equal(refused.status, 2);
  assert.match(refused.stderr, /WORLD_RUN_HOLDS_LOCK=1/);
  const help = spawnSync('bash', [script, '--help'], { encoding: 'utf8', env });
  assert.equal(help.status, 0, help.stderr);
  assert.match(help.stdout, /usage: run-world\.sh/);
  // --print-env is exercised by every test above without the lock.
  const envRun = spawnSync('bash', [script, '--print-env'], { encoding: 'utf8', env });
  assert.equal(envRun.status, 0, envRun.stderr);
});

test('a run refuses to start when the logs drive is below the minimum', (t) => {
  const dir = tempDir(t);
  fs.writeFileSync(path.join(dir, 'victory.toml'), '');
  const result = spawnSync('bash', [script, '--tag', 'lowdisk', '--no-record'], {
    encoding: 'utf8',
    env: isolatedEnv({
      VICTORY_BIN: spawnSync('bash', ['-c', 'command -v bash'], { encoding: 'utf8' }).stdout.trim(),
      VICTORY_DIR: dir,
      VICTORY_CONFIG: path.join(dir, 'victory.toml'),
      GAME_EXE: path.join(dir, 'game.exe'),
      VICTORY_CAPTURES: path.join(dir, 'captures'),
      WORLD_RUN_HOLDS_LOCK: '1',
      HARNESS_MIN_FREE_GB: '999999',
    }),
  });
  assert.equal(result.status, 2, result.stderr);
  assert.match(result.stderr, /refused: disk .* below 999999 GB/);
});

test('the heartbeat reads only the log tail, not the whole file', () => {
  const text = fs.readFileSync(script, 'utf8');
  assert.match(text, /HEARTBEAT_TAIL_BYTES=65536/);
  assert.match(text, /tail -c "\$HEARTBEAT_TAIL_BYTES" "\$LOG"/);
  assert.equal(/grep -aE ' fps, [0-9.]+ ms\/frame' "\$LOG"/.test(text), false, 'not the whole log');
});

test('game output goes to the log only; the terminal goes through the public filter', () => {
  const text = fs.readFileSync(script, 'utf8');
  assert.equal(/tee -a "\$LOG"/.test(text), false, 'no tee of the game output');
  assert.equal(text.includes('tail -2 "$LOG"'), false, 'a failed save reports the log, not its lines');
  assert.match(text, /\. "\$HERE\/public-lines\.sh"/);
  assert.match(text, /"\$BIN" run "\$GAME_EXE" --execute --console "\$CONSOLE"\) \\\n\s+>>"\$LOG" 2>&1/);
  assert.match(text, /--out "\$CAPTURE\/gputrace" \\\n\s+>>"\$LOG" 2>&1/);
  assert.match(text, /--out "\$CAPTURE\/rdoc" --after-ms "\$CAPTURE_AFTER_MS" \\\n\s+>>"\$LOG" 2>&1/);
  for (const line of ['log:', 'capture:', 'profile:'])
    assert.match(text, new RegExp(`say "${line}`), line);
  assert.match(text, /cat "\$LOG\.md" \| public_line/);
  assert.match(text, /heartbeat_loop &/);
  // The header records values, not private paths.
  assert.match(text, /harness_env_line "\$E"/);
  assert.match(text, /harness_env_line "\$NAME=\$\{!NAME\}"/);
  assert.match(text, /capture: \$\(harness_public_path "\$CAPTURE"\)/);
});

test('the warm-up exit code and seconds are written to the run log, not only the terminal', () => {
  const text = fs.readFileSync(script, 'utf8');
  assert.match(text, /WARM_LINE="warm-up: exit=\$WARM_RC after /);
  assert.match(text, /printf '%s\\n' "\$WARM_LINE" >>"\$LOG"/);
  assert.match(text, /\n  say "\$WARM_LINE"\n/);
});

test('the load is sampled before the game starts and after it ends', () => {
  const text = fs.readFileSync(script, 'utf8');
  const start = text.indexOf('load-line.mjs" start');
  const game = text.indexOf('"$BIN" run "$GAME_EXE"');
  const end = text.indexOf('load-line.mjs" end');
  assert.ok(start > 0, 'the start load line is written');
  assert.ok(game > start, 'the start load is taken before the game starts');
  assert.ok(end > game, 'the end load is taken after the game ends');
  assert.match(text, /printf 'harness: load start exit=%s\\n' "\$LOAD_START_RC"/);
  assert.match(text, /printf 'harness: load end exit=%s\\n' "\$\?"/);
});

test('--require-quiet refuses a busy start with exit 3 and is off by default', () => {
  const text = fs.readFileSync(script, 'utf8');
  assert.match(text, /--require-quiet\) REQUIRE_QUIET=1/);
  assert.match(text, /REQUIRE_QUIET=0/);
  assert.match(text, /load-line\.mjs" check "\$LOAD_START"/);
  assert.match(text, /say_err "refused: \$\{QUIET_REASON/);
  assert.match(text, /exit 3/);
  assert.match(text, /\[--require-quiet\]/);
});

test('--quiet-wait sets the wait ceiling, 600 seconds by default', () => {
  const text = fs.readFileSync(script, 'utf8');
  assert.match(text, /QUIET_WAIT=600/);
  assert.match(text, /--quiet-wait\) QUIET_WAIT=\$2/);
  assert.match(text, /--quiet-wait needs a whole number/);
  assert.match(text, /\[--quiet-wait N\]/);
  assert.match(text, /load-line\.mjs" wait --quiet-wait "\$QUIET_WAIT"/);
  assert.match(text, /grep '\^harness: waiting for quiet:' \| public_line/);
});

test('--require-quiet samples before the warm-up and again on the start load', () => {
  const text = fs.readFileSync(script, 'utf8');
  const pre = text.indexOf('quiet_sample\n  PRE_QUIET_RC=$?');
  const warm = text.indexOf('if [ "$WARM" = 1 ]; then');
  const post = text.indexOf('quiet_sample\n  LOAD_START_RC=$?');
  const game = text.indexOf('"$BIN" run "$GAME_EXE"');
  assert.ok(pre > 0, 'the pre-warm-up sample is there');
  assert.ok(pre < warm, 'it runs before the warm-up');
  assert.ok(post > warm, 'the start load is sampled after the warm-up');
  assert.ok(post < game, 'the start load is sampled before the game starts');
});

test('a refusal before the warm-up costs no warm-up', (t) => {
  const dir = tempDir(t);
  fs.writeFileSync(path.join(dir, 'victory.toml'), '');
  const marked = path.join(dir, 'warm-ran');
  const bin = path.join(dir, 'victory-fake');
  fs.writeFileSync(bin, '#!/bin/sh\n: > "$(dirname "$0")/warm-ran"\nexit 0\n');
  fs.chmodSync(bin, 0o755);
  const result = spawnSync(
    'bash',
    [script, '--warm', '--require-quiet', '--quiet-wait', '0', '--no-record', '--volume', 'off', '--tag', 'quiet-refusal'],
    {
      encoding: 'utf8',
      timeout: 300_000,
      env: isolatedEnv({
        VICTORY_BIN: bin,
        VICTORY_DIR: dir,
        VICTORY_CONFIG: path.join(dir, 'victory.toml'),
        GAME_EXE: path.join(dir, 'game.exe'),
        VICTORY_CAPTURES: path.join(dir, 'captures'),
        HARNESS_LOG_DIR: path.join(dir, 'logs'),
        WORLD_RUN_HOLDS_LOCK: '1',
        HARNESS_MIN_FREE_MEM_GB: '999999',
        // The free-space floor off: the machine's own disk (a CI runner's C: has under 30 GB) must not refuse first.
        HARNESS_MIN_FREE_GB: '0',
      }),
    },
  );
  assert.equal(result.status, 3, result.stderr);
  assert.match(result.stderr, /refused: .*free memory .* is under 999999 GB/);
  assert.equal(fs.existsSync(marked), false, 'the warm-up did not run before the refusal');
});

test('the run lock is VICTORY_RUN_LOCK in all three scripts, defaulting to the old path', () => {
  const scripts = {
    'world-run.sh': /LOCK=\$\{VICTORY_RUN_LOCK:-\$HERE\/run\.lock\}/,
    'wait-for-run.sh': /LOCK=\$\{VICTORY_RUN_LOCK:-\$\{HARNESS_RUN_LOCK:-\$HERE\/run\.lock\}\}/,
    'run-world.sh': /LOCK=\$\{VICTORY_RUN_LOCK:-\$TOOLS\/run\.lock\}/,
  };
  for (const [name, pattern] of Object.entries(scripts)) {
    const text = fs.readFileSync(path.join(here, name), 'utf8');
    assert.match(text, pattern, `${name} honours VICTORY_RUN_LOCK`);
  }
  assert.match(fs.readFileSync(script, 'utf8'), /--lock "\$LOCK"/, 'run-world.sh gives its lock to the save restore');
});

test('each helper exit is recorded, and the summary is renamed into place', () => {
  const text = fs.readFileSync(script, 'utf8');
  for (const step of ['disk-usage', 'feed', 'driver-events', 'summary']) {
    assert.match(text, new RegExp(`harness: ${step} exit=`), step);
  }
  assert.match(text, /summarize-run\.mjs" "\$LOG" >"\$LOG\.md\.tmp"/);
  assert.match(text, /mv -f "\$LOG\.md\.tmp" "\$LOG\.md"/);
  assert.doesNotMatch(text, /driver-events\.ps1.*2>\/dev\/null/);
  assert.doesNotMatch(text, /disk-usage\.mjs[^\n]*\|\| true/);
  assert.doesNotMatch(text, /precompile-progress\.mjs" final[^\n]*\|\| true/);
});

test('both scripts are one main() read whole before it runs', () => {
  for (const name of ['run-world.sh', 'world-run.sh']) {
    const text = fs.readFileSync(path.join(here, name), 'utf8');
    assert.match(text, /^main\(\) \{$/m, `${name} wraps its body in main()`);
    assert.match(text, /^main "\$@"$/m, `${name} calls main with the arguments`);
  }
});

test('--live creates an empty control file in the run capture folder and passes its path', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'run-world-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const bash = spawnSync('bash', ['-c', 'command -v bash'], { encoding: 'utf8' }).stdout.trim();
  fs.writeFileSync(path.join(dir, 'victory.toml'), '');
  const result = spawnSync('bash', [script, '--live', '--print-env'], {
    encoding: 'utf8',
    env: isolatedEnv({
      VICTORY_BIN: bash,
      VICTORY_DIR: dir,
      VICTORY_CONFIG: path.join(dir, 'victory.toml'),
      GAME_EXE: path.join(dir, 'game.exe'),
      VICTORY_CAPTURES: path.join(dir, 'captures'),
    }),
  });
  assert.equal(result.status, 0, result.stderr);
  const liveLine = result.stdout.split(/\r?\n/).find((line) => line.startsWith('VICTORY_LIVE='));
  assert.ok(liveLine, 'the runtime receives VICTORY_LIVE');
  assert.match(liveLine, /\/live\.txt$/, 'the control file is in the capture folder');
  assert.match(result.stdout, /^live:\s+live\.txt$/m, 'the terminal sees the file name only');

  const captures = path.join(dir, 'captures');
  const folders = fs.readdirSync(captures);
  assert.equal(folders.length, 1);
  const control = path.join(captures, folders[0], 'live.txt');
  assert.equal(fs.existsSync(control), true, 'the control file is created before the run');
  assert.equal(fs.statSync(control).size, 0, 'it starts empty');
});

// presentmon-compare needs the run's stats over a window (the stats file keeps only its last object) and the game's
// process id (PresentMon's rows are never pooled across processes; the check of 2026-11-26).
test('--live keeps every stats object in stats.jsonl, and every run writes the priority line, with the pid, to the log', () => {
  const text = fs.readFileSync(script, 'utf8');
  assert.match(text, /LIVE_HISTORY=\$\{CAPTURE%\/\}\/stats\.jsonl/);
  assert.match(text, /rm -f "\$LIVE_STATS" "\$LIVE_HISTORY"/);
  assert.match(text, /node "\$TOOLS\/stats-recorder\.mjs" "\$LIVE_STATS" "\$LIVE_HISTORY" --log "\$LOG" >>"\$LOG" 2>&1 &/);
  const recorder = text.indexOf('stats-recorder.mjs"');
  const game = text.indexOf('"$BIN" run "$GAME_EXE"');
  assert.ok(recorder > 0 && recorder < game, 'the recorder starts before the game');
  assert.ok(text.indexOf('kill -0 "$STATS_REC_PID"') > game, 'and is waited for after it');
  assert.match(text, /PRIORITY_LINE=\$\(RUN_WORLD_GAME_EXE=.*raise-priority\.ps1" 2>&1\)/);
  assert.match(text, /printf '%s\\n' "\$PRIORITY_LINE" >>"\$LOG"/);
});

test('--live writes the header line with the control file name, as the other switches do', () => {
  const text = fs.readFileSync(script, 'utf8');
  assert.match(text, /--live\) LIVE=1/);
  assert.match(text, /LIVE_FILE=\$\{CAPTURE%\/\}\/live\.txt/);
  assert.match(text, /: >"\$LIVE_FILE"/);
  assert.match(text, /ENV_ARGS\+=\("VICTORY_LIVE=\$LIVE_FILE"\)/);
  assert.match(text, /say "live: +\$\(basename "\$LIVE_FILE"\)"/);
  assert.match(text, /echo "harness env: live=\$\(basename "\$LIVE_FILE"\)"/);
  assert.match(text, /\[--live\]/);
});

test('--gpu-trace prints the Nsight command it would run, and the run starts the game under it', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'run-world-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const bash = spawnSync('bash', ['-c', 'command -v bash'], { encoding: 'utf8' }).stdout.trim();
  fs.writeFileSync(path.join(dir, 'victory.toml'), '');
  const result = spawnSync('bash', [script, '--gpu-trace', '--trace-after-ms', '330000', '--print-env'], {
    encoding: 'utf8',
    env: isolatedEnv({
      VICTORY_BIN: bash,
      VICTORY_DIR: dir,
      VICTORY_CONFIG: path.join(dir, 'victory.toml'),
      GAME_EXE: path.join(dir, 'game.exe'),
      VICTORY_CAPTURES: path.join(dir, 'captures'),
      VICTORY_NGFX: 'tools/ngfx.exe',
    }),
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /gpu-trace: tools\/ngfx\.exe --activity "GPU Trace Profiler" --exe /);
  assert.match(result.stdout, /--start-after-ms 330000/);
  assert.match(result.stdout, /--output-dir .*gputrace/);
});

// PresentMon is the one frame-time count independent of the runtime's meter (the audit of 2026-11-26 found it never
// ran, silently). The wrapper says at start when it cannot run, and never prints the executable's path.
test('the wrapper says at start when PresentMon is not configured, and does not start its watcher', (t) => {
  const NOT_CONFIGURED = 'PresentMon: not configured (set PRESENTMON_EXE in harness/local.env)';
  const unset = printEnv(t, [], { PRESENTMON_EXE: '' });
  assert.equal(unset.has(NOT_CONFIGURED), true, [...unset].join('\n'));
  const asked = printEnv(t, ['--presentmon'], { PRESENTMON_EXE: '' });
  assert.equal(asked.has(NOT_CONFIGURED), true);
  assert.ok([...asked].some((line) => /^profile: .*presentmon: 0$/.test(line)), 'a watcher that cannot run is not started');

  const missing = printEnv(t, ['--presentmon'], { PRESENTMON_EXE: path.join(tempDir(t), 'PresentMon.exe') });
  assert.equal(missing.has('PresentMon: not configured (PRESENTMON_EXE names no file; set it in harness/local.env)'), true);
  assert.ok([...missing].some((line) => /^profile: .*presentmon: 0$/.test(line)));

  const dir = tempDir(t);
  const exe = path.join(dir, 'PresentMon.exe');
  fs.writeFileSync(exe, '');
  const configured = printEnv(t, ['--presentmon'], { PRESENTMON_EXE: exe });
  assert.equal([...configured].some((line) => line.startsWith('PresentMon: not configured')), false);
  assert.ok([...configured].some((line) => /^profile: .*presentmon: 1$/.test(line)));
  for (const env of [unset, asked, missing, configured]) {
    assert.equal([...env].some((line) => line.includes(dir)), false, 'the executable path is never printed');
  }
});

// PresentMon runs by default once it is configured (the coordinator, 2026-11-26): --no-presentmon turns it off, and
// --presentmon is still accepted.
test('PresentMon runs by default when configured; --no-presentmon turns it off and --presentmon is still accepted', (t) => {
  const dir = tempDir(t);
  const exe = path.join(dir, 'PresentMon.exe');
  fs.writeFileSync(exe, '');
  const header = (env) => [...env].find((line) => line.startsWith('profile: '));
  assert.match(header(printEnv(t, [], { PRESENTMON_EXE: exe })), /presentmon: 1$/, 'on by default');
  assert.match(header(printEnv(t, ['--presentmon'], { PRESENTMON_EXE: exe })), /presentmon: 1$/);
  const off = printEnv(t, ['--no-presentmon'], { PRESENTMON_EXE: exe });
  assert.match(header(off), /presentmon: 0$/, '--no-presentmon turns it off');
  assert.equal([...off].some((line) => line.startsWith('PresentMon: not configured')), false);
  assert.match(header(printEnv(t, [], { PRESENTMON_EXE: '' })), /presentmon: 0$/, 'unconfigured is off');
});
