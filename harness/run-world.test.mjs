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
function printEnvResult(t, args, extraEnv = {}) {
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
  return result;
}

function printEnv(t, args, extraEnv = {}) {
  const result = printEnvResult(t, args, extraEnv);
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

// --settings: every run starts with a named set of switches, so none is measured at bare defaults by accident.
const settingsSwitches = (name) =>
  fs
    .readFileSync(path.join(here, 'settings', `${name}.env`), 'utf8')
    .split(/\r?\n/)
    .filter((line) => /^VICTORY_[A-Z0-9_]+=/.test(line));

test("a run starts with the base build's settings by default, every switch of the file in its env", (t) => {
  const env = printEnv(t, []);
  const switches = settingsSwitches('noone');
  assert.equal(switches.length > 0, true);
  assert.equal(env.has(`harness env: settings=noone (${switches.length} switches)`), true);
  for (const line of switches) assert.equal(env.has(line), true, line);
});

test('--settings none applies no switch from a file', (t) => {
  const env = printEnv(t, ['--settings', 'none']);
  assert.equal(env.has('harness env: settings=none (0 switches)'), true);
  for (const line of settingsSwitches('noone')) assert.equal(env.has(line), false, line);
});

test('a switch given on the command line wins over the settings file', (t) => {
  const env = printEnv(t, [], { VICTORY_CARD_STEPS: '0' });
  const switches = settingsSwitches('noone');
  assert.equal(env.has('VICTORY_CARD_STEPS=2'), false, 'the file must not override the command line');
  assert.equal(env.has(`harness env: settings=noone (${switches.length - 1} switches)`), true);
});

test('--settings refuses a set with no file, and a name that is not a plain word', (t) => {
  for (const name of ['nosuch', '../noone', 'noone.env']) {
    const result = printEnvResult(t, ['--settings', name]);
    assert.equal(result.status, 2, name);
    assert.match(result.stderr + result.stdout, /--settings:/);
  }
});

// --menu-file: the F1 menu's saved choices, applied the way the build's own launcher applies them. The harness
// starts `run`, which hands the menu no file and applies none, so without the option the menu could not save.
function menuFile(t, lines) {
  const file = path.join(tempDir(t), 'victory-settings.txt').replace(/\\/g, '/');
  fs.writeFileSync(file, ['# Saved by the F1 menu', ...lines, ''].join('\r\n'));
  return file;
}
const named = (env, name) => [...env].filter((line) => line.startsWith(`${name}=`));

test('a menu file goes over the settings set: a value replaces, an empty value leaves the switch out', (t) => {
  const set = settingsSwitches('noone');
  assert.equal(set.some((line) => line.startsWith('VICTORY_AMBIENT=')), true, 'the set has the switch this unsets');
  const file = menuFile(t, [
    'VICTORY_RT_RESOLVE=1',
    'VICTORY_ACTIVATE_RAYS=1',
    'VICTORY_AMBIENT=',
    'VICTORY_MENU_DLSS=quality',
    'VICTORY_MENU_FRAMEGEN=2',
  ]);
  const env = printEnv(t, ['--menu-file', file]);
  assert.equal(env.has('VICTORY_RT_RESOLVE=1'), true);
  assert.equal(env.has('VICTORY_ACTIVATE_RAYS=1'), true);
  assert.deepEqual(named(env, 'VICTORY_AMBIENT'), [], 'left out, as the menu saved it');
  assert.equal(env.has('harness env: menu-file=victory-settings.txt (3 values)'), true);
  assert.equal(env.has(`harness env: settings=noone (${set.length - 1} switches)`), true);
  // The menu is handed its file, and is told what the run really has: no folder for DLSS, so neither it nor frame
  // generation is applied.
  assert.deepEqual(named(env, 'VICTORY_MENU_FILE'), [`VICTORY_MENU_FILE=${file}`]);
  assert.equal(env.has('VICTORY_MENU_DLSS=off'), true);
  assert.equal(env.has('VICTORY_MENU_FRAMEGEN=0'), true);
  assert.deepEqual(named(env, 'VICTORY_DLSS'), []);
});

test('a menu file that asks for DLSS turns it on only with a folder that holds the interposer', (t) => {
  const file = menuFile(t, ['VICTORY_MENU_DLSS=quality']);
  const folder = tempDir(t).replace(/\\/g, '/');
  // No folder named, then a folder without the files: off both times, and the run says why.
  for (const extra of [{}, { STREAMLINE_DIR: folder }]) {
    const result = printEnvResult(t, ['--menu-file', file], extra);
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /VICTORY_MENU_DLSS=off/);
    assert.doesNotMatch(result.stdout, /^VICTORY_DLSS=/m);
  }
  fs.writeFileSync(path.join(folder, 'sl.interposer.dll'), '');
  const on = printEnv(t, ['--menu-file', file], { STREAMLINE_DIR: folder });
  assert.equal(on.has('VICTORY_DLSS=1'), true);
  assert.equal(on.has('VICTORY_DLSS_MODE=quality'), true);
  assert.deepEqual(named(on, 'VICTORY_STREAMLINE_DIR'), [`VICTORY_STREAMLINE_DIR=${folder}`]);
  assert.equal(on.has('VICTORY_MENU_DLSS=quality'), true, 'the menu is told what the run really has');
  assert.equal(on.has('VICTORY_MENU_FRAMEGEN=0'), true, 'the menu asks for no frame generation');
  // The command line wins: DLSS switched off there stays off, and the menu shows it.
  const off = printEnv(t, ['--menu-file', file], { STREAMLINE_DIR: folder, VICTORY_DLSS: '0' });
  assert.equal(off.has('VICTORY_MENU_DLSS=off'), true);
  // A mode the build does not know is ignored.
  const odd = printEnv(t, ['--menu-file', menuFile(t, ['VICTORY_MENU_DLSS=ultra'])], { STREAMLINE_DIR: folder });
  assert.deepEqual(named(odd, 'VICTORY_DLSS'), []);
});

// Frame generation, as the build's own launcher sets it: over DLSS with NVIDIA's files, and the line of submissions.
const fgSwitches = ['VICTORY_FRAMEGEN', 'VICTORY_FRAMEGEN_X', 'VICTORY_TEXTURE_ROOM_PERCENT', 'VICTORY_FSR_FRAMEGEN'];
function streamlineFolder(t) {
  const folder = tempDir(t).replace(/\\/g, '/');
  fs.writeFileSync(path.join(folder, 'sl.interposer.dll'), '');
  return folder;
}
function lines(result) {
  assert.equal(result.status, 0, result.stderr);
  return new Set(result.stdout.split(/\r?\n/).map((line) => line.trim()).filter(Boolean));
}
function assertFgOff(env, why) {
  for (const name of fgSwitches) assert.deepEqual(named(env, name), [], name);
  assert.equal(env.has('VICTORY_MENU_FRAMEGEN=0'), true, 'the menu is told frame generation is off');
  if (why) assert.equal([...env].some((line) => line.startsWith('menu file: ') && why.test(line)), true, String(why));
}

test('a menu file that asks for frame generation turns it on at its factor over DLSS with the folder', (t) => {
  const folder = streamlineFolder(t);
  for (const type of [[], ['VICTORY_MENU_FG_TYPE=dlss'], ['VICTORY_MENU_FG_TYPE=auto']]) {
    const file = menuFile(t, ['VICTORY_MENU_DLSS=quality', 'VICTORY_MENU_FRAMEGEN=3', ...type]);
    const env = lines(printEnvResult(t, ['--menu-file', file], { STREAMLINE_DIR: folder }));
    assert.equal(env.has('VICTORY_DLSS=1'), true);
    assert.deepEqual(named(env, 'VICTORY_FRAMEGEN'), ['VICTORY_FRAMEGEN=1'], type.join());
    assert.deepEqual(named(env, 'VICTORY_FRAMEGEN_X'), ['VICTORY_FRAMEGEN_X=3']);
    assert.deepEqual(named(env, 'VICTORY_TEXTURE_ROOM_PERCENT'), ['VICTORY_TEXTURE_ROOM_PERCENT=80']);
    assert.deepEqual(named(env, 'VICTORY_FSR_FRAMEGEN'), [], 'DLSS frame generation, not FSR 3');
    assert.deepEqual(named(env, 'VICTORY_MENU_FRAMEGEN'), ['VICTORY_MENU_FRAMEGEN=3'], 'the menu is told the run has it');
    assert.equal(env.has('harness env: menu-file=victory-settings.txt (6 values)'), true, 'three for DLSS, three for it');
    assert.equal([...env].some((line) => /^menu file: .*frame generation 3x \(DLSS\)/.test(line)), true);
  }
});

test('frame generation stays off without DLSS and its folder, and the run says why', (t) => {
  const empty = tempDir(t).replace(/\\/g, '/');
  const folder = streamlineFolder(t);
  for (const [menu, extra] of [
    [['VICTORY_MENU_DLSS=quality'], {}],
    [['VICTORY_MENU_DLSS=quality'], { STREAMLINE_DIR: empty }],
    [['VICTORY_MENU_DLSS=off'], { STREAMLINE_DIR: folder }],
    [[], { STREAMLINE_DIR: folder }],
  ]) {
    const file = menuFile(t, [...menu, 'VICTORY_MENU_FRAMEGEN=2']);
    assertFgOff(lines(printEnvResult(t, ['--menu-file', file], extra)), /needs DLSS upscaling on with NVIDIA's files/);
  }
});

test('a VICTORY_FRAMEGEN=0 on the command line keeps frame generation off whatever the menu says', (t) => {
  const folder = streamlineFolder(t);
  for (const type of ['dlss', 'fsr']) {
    const file = menuFile(t, ['VICTORY_MENU_DLSS=quality', 'VICTORY_MENU_FRAMEGEN=4', `VICTORY_MENU_FG_TYPE=${type}`]);
    const env = lines(printEnvResult(t, ['--menu-file', file], { STREAMLINE_DIR: folder, VICTORY_FRAMEGEN: '0' }));
    assert.equal(env.has('VICTORY_DLSS=1'), true, 'DLSS itself is still applied');
    assertFgOff(env, /VICTORY_FRAMEGEN=0 keeps it off/);
  }
});

test('a frame generation factor outside 2 to 6 is ignored', (t) => {
  const folder = streamlineFolder(t);
  for (const factor of ['7', '1', 'abc', '']) {
    const file = menuFile(t, ['VICTORY_MENU_DLSS=quality', `VICTORY_MENU_FRAMEGEN=${factor}`]);
    const env = lines(printEnvResult(t, ['--menu-file', file], { STREAMLINE_DIR: folder }));
    assertFgOff(env);
    assert.equal([...env].some((line) => /frame generation \(/.test(line)), false, `no reason line for '${factor}'`);
  }
});

test("frame generation needs VICTORY_THIN=1 or VICTORY_ASYNC_SUBMIT=1 in the run's final switches", (t) => {
  const folder = streamlineFolder(t);
  const file = menuFile(t, ['VICTORY_MENU_DLSS=quality', 'VICTORY_MENU_FRAMEGEN=2']);
  const why = /needs VICTORY_THIN=1 or VICTORY_ASYNC_SUBMIT=1/;
  // The set has VICTORY_THIN=1; an empty value on the command line takes it out of the run, and so does the menu file.
  const bare = { STREAMLINE_DIR: folder, VICTORY_THIN: '', VICTORY_JOIN_PARTS: '' };
  const offCmd = lines(printEnvResult(t, ['--menu-file', file], bare));
  assert.deepEqual(named(offCmd, 'VICTORY_THIN'), [], "the command line took the set's switch out");
  assertFgOff(offCmd, why);
  const unset = menuFile(t, ['VICTORY_THIN=', 'VICTORY_MENU_DLSS=quality', 'VICTORY_MENU_FRAMEGEN=2']);
  assertFgOff(lines(printEnvResult(t, ['--menu-file', unset], { STREAMLINE_DIR: folder })), why);
  assertFgOff(lines(printEnvResult(t, ['--menu-file', file, '--settings', 'none'], { STREAMLINE_DIR: folder })), why);
  // VICTORY_ASYNC_SUBMIT=1 given instead: on.
  const on = lines(printEnvResult(t, ['--menu-file', file], { ...bare, VICTORY_ASYNC_SUBMIT: '1' }));
  assert.deepEqual(named(on, 'VICTORY_FRAMEGEN'), ['VICTORY_FRAMEGEN=1']);
  assert.deepEqual(named(on, 'VICTORY_FRAMEGEN_X'), ['VICTORY_FRAMEGEN_X=2']);
  assert.equal(on.has('VICTORY_MENU_FRAMEGEN=2'), true);
});

test("the menu's FSR 3 type puts VICTORY_FSR_FRAMEGEN=1 in place of VICTORY_FRAMEGEN", (t) => {
  const folder = streamlineFolder(t);
  const file = menuFile(t, ['VICTORY_MENU_DLSS=balanced', 'VICTORY_MENU_FRAMEGEN=2', 'VICTORY_MENU_FG_TYPE=fsr']);
  const env = lines(printEnvResult(t, ['--menu-file', file], { STREAMLINE_DIR: folder }));
  assert.deepEqual(named(env, 'VICTORY_FSR_FRAMEGEN'), ['VICTORY_FSR_FRAMEGEN=1']);
  assert.deepEqual(named(env, 'VICTORY_FRAMEGEN'), [], 'FSR only, as the launcher sets it');
  assert.deepEqual(named(env, 'VICTORY_FRAMEGEN_X'), ['VICTORY_FRAMEGEN_X=2']);
  assert.deepEqual(named(env, 'VICTORY_TEXTURE_ROOM_PERCENT'), ['VICTORY_TEXTURE_ROOM_PERCENT=80']);
  assert.equal(env.has('VICTORY_MENU_FRAMEGEN=2'), true);
  assert.equal(env.has('VICTORY_MENU_FG_TYPE=fsr'), true);
  assert.equal([...env].some((line) => /^menu file: .*frame generation 2x \(FSR 3\)/.test(line)), true);
  // Without DLSS's folder FSR 3 frame generation stays off too.
  assertFgOff(lines(printEnvResult(t, ['--menu-file', file], {})), /needs DLSS upscaling on/);
});

test("the menu's launcher choices become the switches the build's own launcher makes of them", (t) => {
  const env = printEnv(t, ['--menu-file', menuFile(t, ['VICTORY_MENU_RES=1440', 'VICTORY_MENU_FPS=60'])]);
  assert.equal(env.has('VICTORY_NATIVE_1440P=1'), true);
  assert.deepEqual(named(env, 'VICTORY_FRAME_INTERVAL_US'), ['VICTORY_FRAME_INTERVAL_US=16667'], "over the set's");
  assert.equal(env.has('VICTORY_MENU_RES=1440'), true);
  assert.equal(env.has('VICTORY_MENU_FPS=60'), true);

  const stretched = printEnv(t, ['--menu-file', menuFile(t, ['VICTORY_MENU_RES=1080to1440'])]);
  assert.equal(stretched.has('VICTORY_RENDER_1080P=1'), true);
  assert.equal(stretched.has('VICTORY_MENU_RES=1080to1440'), true);
});

test('a menu file that does not exist yet still lets the menu save, and shows the stock picture', (t) => {
  const file = path.join(tempDir(t), 'not-yet.txt').replace(/\\/g, '/');
  const env = printEnv(t, ['--menu-file', file]);
  assert.deepEqual(named(env, 'VICTORY_MENU_FILE'), [`VICTORY_MENU_FILE=${file}`]);
  assert.equal(env.has('VICTORY_MENU_RES=4k'), true);
  assert.equal(env.has('VICTORY_MENU_FPS=1000'), true);
  assert.equal(env.has('VICTORY_MENU_CONSOLE=series-x'), true);
  assert.equal(env.has('harness env: menu-file=not-yet.txt (0 values)'), true);
});

test('the command line and --console win over the menu file', (t) => {
  const file = menuFile(t, ['VICTORY_RT_RESOLVE=1', 'VICTORY_MENU_CONSOLE=series-s', 'VICTORY_MENU_RES=1440']);
  const menu = printEnv(t, ['--menu-file', file]);
  assert.equal(menu.has('VICTORY_MENU_CONSOLE=series-s'), true, "the menu's console when none is given");
  assert.deepEqual(named(menu, 'VICTORY_NATIVE_1440P'), [], 'no 1440p switch on the smaller console, as in the launcher');
  const given = printEnv(t, ['--menu-file', file, '--console', 'series-x'], { VICTORY_RT_RESOLVE: '0' });
  assert.equal(given.has('VICTORY_MENU_CONSOLE=series-x'), true);
  assert.deepEqual(named(given, 'VICTORY_RT_RESOLVE'), [], 'the menu must not override the command line');
  assert.equal(given.has('VICTORY_NATIVE_1440P=1'), true);
});

test('without --menu-file nothing about the menu is passed', (t) => {
  for (const line of printEnv(t, [])) assert.doesNotMatch(line, /^VICTORY_MENU_|menu-file/);
});

// --boot-video: the loading cover, a video over the screen from the start until the world is on it.
test('--boot-video names its file in the env header, and --boot-volume the level the game is heard at meanwhile', (t) => {
  const video = path.join(tempDir(t), 'boot.mp4').replace(/\\/g, '/');
  fs.writeFileSync(video, '');
  const env = printEnv(t, ['--boot-video', video]);
  assert.equal(env.has('harness env: boot-video=boot.mp4'), true);
  const heard = printEnv(t, ['--boot-video', video, '--boot-volume', '30']);
  assert.equal(heard.has('harness env: boot-video=boot.mp4 (the game at 30 while it shows)'), true);
  for (const line of printEnv(t, [])) assert.doesNotMatch(line, /boot-video/);
});

test('--boot-video refuses a file that is not there; --boot-volume refuses to stand alone or to be a word', (t) => {
  const video = path.join(tempDir(t), 'boot.mp4').replace(/\\/g, '/');
  fs.writeFileSync(video, '');
  for (const args of [
    ['--boot-video', `${video}.missing`],
    ['--boot-volume', '30'],
    ['--boot-video', video, '--boot-volume', 'loud'],
  ]) {
    const result = printEnvResult(t, args);
    assert.equal(result.status, 2, args.join(' '));
    assert.match(result.stderr + result.stdout, /--boot-(video|volume)/);
  }
});

test('the loading cover starts before the game, is watched from the log, and ends with the run', () => {
  const text = fs.readFileSync(script, 'utf8');
  const player = text.indexOf('ffplay -loglevel quiet -an -loop 0 -fs -alwaysontop');
  const watcher = text.indexOf('node "$TOOLS/boot-cover.mjs" watch "$LOG" "$COVER_PID"');
  const game = text.indexOf('"$BIN" run "$GAME_EXE" --execute --console "$CONSOLE") \\');
  assert.ok(player > 0 && watcher > player, 'the player, then its watcher');
  assert.ok(game > watcher, 'both before the game starts, so its window opens under the cover');
  assert.match(text, / -an /, "the video's own sound is never played");
  assert.match(text, /trap 'cover_end; pipeline_library_leave' EXIT/, 'the player never outlives the run');
  assert.ok(text.indexOf('\ncover_end\n', game) > game, 'and it is ended when the game ends');
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

test("the warm-up gets the settings set's switches and uses the build's own translate command", () => {
  const text = fs.readFileSync(script, 'utf8');
  const warm = text.indexOf('if [ "$WARM" = 1 ]; then');
  const settingsToWarm = text.indexOf('WARM_ENV+=(${SETTINGS_PAIRS[@]+"${SETTINGS_PAIRS[@]}"})');
  const run = text.indexOf('"$BIN" "$WARM_CMD"');
  assert.ok(text.includes('SETTINGS_PAIRS+=("$PAIR")'), 'each switch taken from the file is kept for the warm-up');
  assert.ok(settingsToWarm > warm, 'the warm-up environment carries them');
  assert.ok(run > settingsToWarm, 'and the command runs with that environment');
  assert.match(text, /"\$BIN" shaders --help >\/dev\/null 2>&1 && WARM_CMD=shaders/, 'shaders where the build has it');
  assert.doesNotMatch(text, /"\$BIN" warm \$\{WARM_ARGS/, 'never the fixed warm command');
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
  // Each waiting line is said as it comes, through the public filter: a wait that prints only when it returns
  // looks like a hang (a run waited 20 minutes in silence on 2026-11-28).
  assert.match(text, /wait --quiet-wait "\$QUIET_WAIT" 2>>"\$LOG" \| while IFS= read -r line; do/);
  assert.match(text, /'harness: waiting for quiet:'\*\)\s+printf '%s\\n' "\$line" \| public_line >&3/);
  assert.match(text, /exit "\$\{PIPESTATUS\[0\]\}"/, "the wait's own exit code is kept");
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

test('--world-timeout is passed through and printed as an environment line', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'run-world-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const bash = spawnSync('bash', ['-c', 'command -v bash'], { encoding: 'utf8' }).stdout.trim();
  fs.writeFileSync(path.join(dir, 'victory.toml'), '');
  const result = spawnSync('bash', [script, '--world-timeout', '300', '--print-env'], {
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
  assert.match(result.stdout, /harness env: world-timeout=300/);
});

// The key routes follow places in the title, so the public tree does not carry them: there these two tests skip, and
// say so. The routes' own handling is tested above with a route the test writes (HARNESS_ROUTES_DIR).
const NO_ROUTES = fs.existsSync(path.join(here, 'routes')) ? false : 'the key routes are not part of this tree';

test('--protocol team-270 sets the route and the settled window, and prints them', { skip: NO_ROUTES }, (t) => {
  const env = printEnv(t, ['--protocol', 'team-270']);
  assert.ok(env.has('harness env: protocol=team-270'), [...env].join('\n'));
  assert.ok(env.has('harness env: route=team-270'));
  assert.ok(env.has('harness env: settled-seconds=275'));
  // The route's entries reach the run through VICTORY_KEYS: the walk phase's first held key is in the list.
  const keys = [...env].find((line) => line.startsWith('VICTORY_KEYS='));
  assert.ok(keys, [...env].join('\n'));
  assert.ok(keys.includes('+150:W:3'), keys);
});

test('--print-env without --protocol is unchanged by it: no protocol, route or settled lines', { skip: NO_ROUTES }, (t) => {
  const env = printEnv(t, ['--route', 'manual-1']);
  assert.ok([...env].some((line) => line.startsWith('VICTORY_KEYS=')), 'the route still reaches the run');
  assert.ok(![...env].some((line) => line.startsWith('harness env: protocol=')), [...env].join('\n'));
  assert.ok(![...env].some((line) => line.startsWith('harness env: route=')));
  assert.ok(![...env].some((line) => line.startsWith('harness env: settled-seconds=')));
});
