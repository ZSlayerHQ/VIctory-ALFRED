// The fixtures against the runtime they were copied from: each one re-read from VIctory's own source (its format
// strings, its constants and the strings its tests pin), so a fixture that has drifted from what `world` writes fails
// here and not in a run. Skipped, with the reason printed, where there is no victory checkout (the public copy).
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import test from 'node:test';

import {
  CHANGE_LINE,
  BUDGET_LINE,
  BUDGET_LINE_OTHERS,
  FRAME_LINE,
  GPU_LINE,
  KEYS_DONE_LINE,
  MARK_LINE,
  SEGMENT_BEGUN_AT_MARK_LINE,
  SEGMENT_CLOSED_LINE,
  SEGMENT_ENDED_BY_MARK_LINE,
  SEGMENT_FIRST_LINE,
  SEGMENT_OPEN_LINE,
  SEGMENT_RUN_END_LINE,
  SEGMENT_TIMER_OPEN_LINE,
} from './runtime-lines.mjs';
import {
  ENTRY_PHASES,
  IN_FORCE_DEFAULTS,
  RUNTIME_PHASES,
  STATS_ALL_OFF,
  STATS_LIVE,
  STATS_OPTIONAL_PARTS,
  statsWith,
} from './runtime-stats.mjs';
import { constants, fnBody, formatRegex, gpuSource, literalWith, literalsIn, loopArrays, scan, victoryRoot } from './rust-source.mjs';

/// The commit of `world` the fixtures were copied at.
const READ_AT = '2f6b742fb';

const root = victoryRoot();

function runtime(t) {
  if (root) return true;
  t.skip('no victory checkout (set VICTORY_REPO): the fixtures cannot be checked against the runtime here');
  console.log('runtime-shapes: skipped, no victory checkout (set VICTORY_REPO)');
  return false;
}

const files = {};
const source = (relative) => (files[relative] ??= gpuSource(root, relative));

/// The one `format!` string of `relative` that `find` picks, as an anchored expression with the file's constants.
function format(relative, find, extra = {}) {
  const src = source(relative);
  const text = literalWith(src, find);
  return formatRegex(text, { ...constants(src), ...extra });
}

function matches(regex, text, what) {
  const m = regex.re.exec(text);
  assert.ok(m, `${what} does not have the shape of the runtime's format string: ${text.slice(0, 160)}...`);
  return m;
}

const keysOf = (jsonText) => Object.keys(JSON.parse(jsonText));

/// The raw text of a top-level key's value in an object's text, by balanced brackets (two arguments side by side in
/// a format string, `{kernel_calls}{optional}`, cannot be told apart by the expression alone).
function valueText(text, key) {
  const at = text.indexOf(`"${key}":`);
  assert.ok(at > 0, `no ${key}`);
  let i = at + key.length + 3;
  const start = i;
  let depth = 0;
  let inString = false;
  for (; i < text.length; i += 1) {
    const c = text[i];
    if (inString) {
      if (c === '\\') i += 1;
      else if (c === '"') inString = false;
    } else if (c === '"') inString = true;
    else if (c === '{' || c === '[') depth += 1;
    else if (c === '}' || c === ']') {
      if (depth === 0) break;
      depth -= 1;
    } else if (c === ',' && depth === 0) break;
  }
  return text.slice(start, i);
}
/// What follows kernel_calls in an object's text: the optional parts.
const optionalText = (text) => {
  const kernel = valueText(text, 'kernel_calls');
  return text.slice(text.indexOf(`"kernel_calls":${kernel}`) + `"kernel_calls":${kernel}`.length, -1);
};

/// The registry's names in its own order and their kinds (live.rs `switches!`).
function registry() {
  const src = source('live.rs');
  const table = src.slice(src.indexOf('switches! {'));
  return [...table.matchAll(/^\s+[A-Z0-9_]+ => "([A-Z0-9_]+)", (OnOff|Integer|List|Keys|Mark),/gm)].map((m) => ({
    name: m[1],
    kind: m[2],
  }));
}

test('the checkout read is the commit the fixtures name, or the checks below say what moved', (t) => {
  if (!runtime(t)) return;
  let head = null;
  try {
    head = execFileSync('git', ['-C', root, 'rev-parse', '--short=9', 'HEAD'], { encoding: 'utf8' }).trim();
  } catch {
    // Not a git checkout: the content checks still run.
  }
  if (head && !head.startsWith(READ_AT)) console.log(`runtime-shapes: fixtures read at world ${READ_AT}, checkout at ${head}`);
  assert.ok(true);
});

test('the phase list is budget.rs NAMES, and the entry phases are its ENTRY_PHASES', (t) => {
  if (!runtime(t)) return;
  const src = source('budget.rs');
  const { masked } = scan(src);
  const at = masked.indexOf('const NAMES: [&str; PHASES] = [');
  const end = masked.indexOf('];', at);
  assert.ok(at > 0, 'NAMES was found');
  assert.deepEqual(literalsIn(src, at, end).map((lit) => lit.value), [...RUNTIME_PHASES]);
  const variants = Object.fromEntries([...masked.matchAll(/^\s+([A-Z][A-Za-z]+) = (\d+),$/gm)].map((m) => [m[1], Number(m[2])]));
  const entry = /const ENTRY_PHASES: \[Phase; \d+\] = \[([^\]]+)\]/.exec(masked)[1];
  const names = [...entry.matchAll(/Phase::([A-Za-z]+)/g)].map((m) => RUNTIME_PHASES[variants[m[1]]]);
  assert.deepEqual(names, [...ENTRY_PHASES]);
});

test('every stats fixture has the shape of stats.rs json, its keys and its optional parts in order', (t) => {
  if (!runtime(t)) return;
  const object = format('host/stats.rs', '"seq":{seq},');
  const budget = format('host/stats.rs', '{{"thread":"presenting"');
  const others = formatRegex(literalWith(source('host/stats.rs'), '', { where: (v) => v.startsWith('{{"scopes":') }));
  const phases = (text) => keysOf(`{${text}}`);
  const all = [['STATS_ALL_OFF', STATS_ALL_OFF], ['STATS_LIVE', STATS_LIVE]];
  for (const name of Object.keys(STATS_OPTIONAL_PARTS)) all.push([`statsWith(${name})`, statsWith(name)]);
  for (const [what, text] of all) {
    const m = matches(object, text, what);
    const b = matches(budget, valueText(text, 'budget'), `${what} budget`);
    assert.deepEqual(phases(b.at(-1)), RUNTIME_PHASES.filter((p) => p !== 'pool_work' && p !== 'relay_land'));
    const o = matches(others, valueText(text, 'other_threads'), `${what} other_threads`);
    assert.deepEqual(phases(o.at(-1)), RUNTIME_PHASES.filter((p) => p !== 'title'));
    for (const [phase, entry] of Object.entries(JSON.parse(valueText(text, 'budget')).phases)) {
      if (ENTRY_PHASES.includes(phase)) assert.equal(entry.cpu_ms, null, `${what}: ${phase} has no processor time of its own`);
    }
    JSON.parse(text);
  }
  // With every switch off, nothing after kernel_calls.
  assert.equal(optionalText(STATS_ALL_OFF), '');
  // The optional parts in optional()'s order, the hash check's fields first.
  const src = source('host/stats.rs');
  const body = fnBody(src, 'optional');
  const objectsAt = scan(src).masked.indexOf('let objects:', body.start);
  const optionalNames = literalsIn(src, objectsAt, body.end).map((lit) => lit.value).filter((v) => /^[a-z_]+$/.test(v));
  assert.deepEqual(Object.keys(STATS_OPTIONAL_PARTS), ['hash_check', ...optionalNames]);
  for (const name of Object.keys(STATS_OPTIONAL_PARTS)) assert.equal(optionalText(statsWith(name)), STATS_OPTIONAL_PARTS[name]);
  matches(format('host/hash_check.rs', ',"hash_missed_pieces_total":'), STATS_OPTIONAL_PARTS.hash_check, 'the hash fields');
  const part = (name) => STATS_OPTIONAL_PARTS[name].slice(`,"${name}":`.length);
  matches(format('receipts.rs', '{{"installs_total":'), part('texture_receipts'), 'texture_receipts');
  matches(format('host/stats.rs', '{{"uploads_total":'), part('texture_recheck'), 'texture_recheck');
  matches(format('host/stats.rs', '{{"new_keys_total":'), part('textures_made'), 'textures_made');
  // landed_classes holds `null` until its first window ends (landed_json).
  const landings = source('memclass/census/landings.rs');
  const landed = fnBody(landings, 'landed_json');
  assert.ok(literalsIn(landings, landed.start, landed.end).some((l) => l.value === 'null'));
  assert.equal(part('landed_classes'), 'null');
});

test("the stats fixtures hold what stats.rs's tests pin, and none of the names they say are gone", (t) => {
  if (!runtime(t)) return;
  const src = source('host/stats.rs');
  for (const test of ['the_object_has_every_field', 'totals_say_total_and_the_worst_frame_is_called_that']) {
    for (const key of loopArrays(src, test, 'key')) assert.ok(STATS_ALL_OFF.includes(key), `${test}: ${key}`);
    for (const gone of loopArrays(src, test, 'gone')) assert.ok(!STATS_ALL_OFF.includes(gone), `${test}: ${gone} is gone`);
  }
  // STATS_LIVE holds a segment_open (its own "segment"), so the names said gone are checked among its top-level keys.
  const top = Object.keys(JSON.parse(STATS_LIVE));
  for (const gone of ['frame_ms_p95', 'segment', 'kernel_calls_total', 'parts', 'dispatches', 'stores_failed', 'pipelines_created']) {
    assert.ok(!top.includes(gone), gone);
  }
  for (const key of loopArrays(src, 'title_is_only_the_title_and_the_calls_are_counted', 'key').filter((k) => k.startsWith('"'))) {
    assert.ok(STATS_LIVE.includes(key), `title_is_only_the_title...: ${key}`);
  }
  const budgetTest = 'the_budget_is_in_the_object_and_other_threads_are_apart';
  for (const find of ['"budget":{"thread":"presenting"', '"relay_wait":{"ms":80.000', '"relay_land":{"ms":60.000']) {
    assert.ok(STATS_ALL_OFF.includes(literalWith(src, find, { fn: budgetTest })), find);
  }
  const rest = STATS_ALL_OFF.split('"other_threads":')[1];
  assert.ok(rest.startsWith(literalWith(src, '{"scopes":2.0,"phases":{"list_calls":', { fn: budgetTest })));
  assert.ok(STATS_ALL_OFF.includes(literalWith(src, '"dispatched":12.0', { fn: 'dispatches_are_per_frame' })));
});

test('the live fixture: its live map is the registry in order, its segment_open and gpu_stamps the runtime tests', (t) => {
  if (!runtime(t)) return;
  const object = JSON.parse(STATS_LIVE);
  const names = registry().map((spec) => spec.name).filter((name) => name !== 'VICTORY_LIVE_HUD');
  assert.ok(names.includes('LIVE_MARK'));
  assert.deepEqual(Object.keys(object.live), names);
  const m = matches(format('host/stats.rs', '"seq":{seq},'), STATS_LIVE, 'STATS_LIVE');
  const open = valueText(STATS_LIVE, 'segment_open');
  matches(format('live.rs', '{{"segment":{k},'), open, 'segment_open');
  assert.equal(open, literalWith(source('live.rs'), '"transition_excluded":3', { fn: 'the_marker_and_segment_lines_have_their_fixed_shape' }));
  const stamps = valueText(STATS_LIVE, 'gpu_stamps');
  matches(format('host/queue_stamps.rs', '{{"copy_queue":'), stamps, 'gpu_stamps');
  const pinned = literalWith(source('host/queue_stamps.rs'), '{"copy_queue":{"fetch_lists":2.00', { fn: 'a_copy_queue_stamp_appears_in_the_report' });
  assert.ok(stamps.startsWith(pinned));
  const kinds = literalsIn(source('host/queue_stamps.rs'), fnBody(source('host/queue_stamps.rs'), 'not_stamped_with').start, fnBody(source('host/queue_stamps.rs'), 'not_stamped_with').end).map((l) => l.value);
  for (const kind of JSON.parse(stamps).not_stamped) assert.ok(kinds.includes(kind), kind);
  assert.ok(stamps.includes(literalWith(source('host/queue_stamps.rs'), '"idle":"not derivable', { fn: 'idle_is_said_not_to_be_derivable' })));
});

test("kernel_calls has kernelstat.rs's shape and names", (t) => {
  if (!runtime(t)) return;
  const src = source('kernelstat.rs');
  const m = matches(format('host/stats.rs', '"seq":{seq},'), STATS_ALL_OFF, 'STATS_ALL_OFF');
  const text = valueText(STATS_ALL_OFF, 'kernel_calls');
  const k = matches(format('kernelstat.rs', '{{"per_frame":{per_frame},'), text, 'kernel_calls');
  matches(format('kernelstat.rs', '{{"reads":{r:.1},'), k.groups.per_frame, 'kernel_calls.per_frame');
  const object = JSON.parse(text);
  const list = (name) => {
    const { masked } = scan(src);
    const at = masked.indexOf(`const ${name}:`);
    return literalsIn(src, at, masked.indexOf('];', at)).map((l) => l.value);
  };
  assert.deepEqual(object.classes, list('CLASS_NAMES'));
  assert.deepEqual(object.read_fields, list('READ_FIELDS'));
  assert.deepEqual(Object.keys(object.protects_by_place_total), list('PLACE_NAMES'));
  assert.deepEqual(Object.keys(object.write_watch_parts_total), list('WRITE_WATCH_NAMES'));
  const callers = literalsIn(src, fnBody(src, 'caller_name').start, fnBody(src, 'caller_name').end).map((l) => l.value);
  assert.deepEqual(Object.keys(object.reads_by_caller_total), callers);
  assert.ok(text.startsWith(literalWith(src, '{"per_frame":{"reads":', { fn: 'the_stats_figures_are_per_frame_or_totals' })));
});

test("every segment line has segment_line's shape and holds what live.rs's tests pin", (t) => {
  if (!runtime(t)) return;
  const src = source('live.rs');
  const shape = format('live.rs', 'live segment {k}{}: frames {first}..{last}');
  for (const [what, line] of Object.entries({
    SEGMENT_OPEN_LINE,
    SEGMENT_CLOSED_LINE,
    SEGMENT_FIRST_LINE,
    SEGMENT_ENDED_BY_MARK_LINE,
    SEGMENT_BEGUN_AT_MARK_LINE,
    SEGMENT_TIMER_OPEN_LINE,
    SEGMENT_RUN_END_LINE,
  })) {
    matches(shape, line, what);
  }
  const figures = (line) => line.split(', skipped ')[0];
  const fixed = 'the_marker_and_segment_lines_have_their_fixed_shape';
  assert.equal(figures(SEGMENT_OPEN_LINE), literalWith(src, 'live segment 2 open: frames 10..19', { fn: fixed }));
  const notEnded = 'not ended (the segment under way, as of this line)';
  assert.equal(figures(SEGMENT_CLOSED_LINE), figures(SEGMENT_OPEN_LINE).replace(' open:', ':').replace(notEnded, 'ended by a change'));
  assert.equal(CHANGE_LINE, literalWith(src, 'live: VICTORY_TEXTURE_LOW 70 -> 60', { fn: fixed }));
  const leaves = 'a_segment_leaves_its_transition_light_and_long_frames_out_and_says_so';
  assert.ok(SEGMENT_FIRST_LINE.includes(literalWith(src, ': frames 1..4, 2 counted', { fn: leaves })));
  const mark = 'a_mark_cuts_a_segment_and_leaves_no_frame_out';
  assert.ok(SEGMENT_ENDED_BY_MARK_LINE.includes(literalWith(src, 'ended by a mark, ', { fn: mark })));
  assert.ok(SEGMENT_BEGUN_AT_MARK_LINE.includes(literalWith(src, ': frames 3..5, 3 counted', { fn: mark })));
  assert.ok(SEGMENT_BEGUN_AT_MARK_LINE.includes(literalWith(src, 'begun at a mark, ended by a change', { fn: mark })));
  const timer = 'a_run_ended_by_the_timer_has_its_last_segment_written';
  assert.ok(SEGMENT_TIMER_OPEN_LINE.startsWith(literalWith(src, 'live segment 2 open: frames 2..7', { fn: timer })));
  assert.ok(SEGMENT_TIMER_OPEN_LINE.includes(literalWith(src, 'mean 240.0, p50 240.0, p95 244.0', { fn: timer })));
  assert.equal(figures(SEGMENT_RUN_END_LINE), figures(SEGMENT_TIMER_OPEN_LINE).replace(' open:', ':').replace(notEnded, 'ended by the end of the run'));
  // The in-force list: the registry's names in order, the panel left out while off, each value of its kind's shape.
  const kinds = Object.fromEntries(registry().map((spec) => [spec.name, spec.kind]));
  const pairs = IN_FORCE_DEFAULTS.split(' ').map((pair) => pair.split('='));
  assert.deepEqual(pairs.map(([name]) => name), registry().map((s) => s.name).filter((n) => n !== 'VICTORY_LIVE_HUD'));
  const shapes = { OnOff: /^(on|off)$/, Integer: /^\d+$/, Mark: /^\d+$/, List: /^$/, Keys: /^$/ };
  for (const [name, value] of pairs) assert.match(value, shapes[kinds[name]], name);
});

test("the change, mark and keys lines have live.rs's shapes; LIVE_MARK is a switch of the kind Mark", (t) => {
  if (!runtime(t)) return;
  const marker = format('live.rs', 'live: {name} {} -> {} at frame {frame}');
  matches(marker, CHANGE_LINE, 'CHANGE_LINE');
  assert.equal(matches(marker, MARK_LINE, 'MARK_LINE').groups.name, 'LIVE_MARK');
  assert.deepEqual(registry().find((spec) => spec.name === 'LIVE_MARK'), { name: 'LIVE_MARK', kind: 'Mark' });
  assert.equal(KEYS_DONE_LINE, literalWith(source('live.rs'), 'live keys: sequence done at frame 1234', { fn: 'the_done_line_has_its_fixed_shape_and_a_run_stops_once' }));
});

test("the frame, budget and GPU lines have stats.rs's and queue_stamps.rs's shapes and what their tests pin", (t) => {
  if (!runtime(t)) return;
  const stats = source('host/stats.rs');
  const frame = matches(format('host/stats.rs', 'frame {frame}, {fps:.1} fps'), FRAME_LINE, 'FRAME_LINE');
  matches(format('host/stats.rs', 'means over frames {}..{}'), frame.groups.window, 'the window clause');
  assert.ok(frame.groups.window.startsWith(literalWith(stats, 'means over frames 1..6', { fn: 'a_frame_line_covers_every_second_since_the_last' })));
  matches(format('host.rs', '{:.1} host waits (for the relay {relay:.1}'), frame.groups.waits, 'the host waits');
  const host = source('host.rs');
  const usage = fnBody(host, 'text', { after: 'impl Usage {' });
  assert.ok(literalsIn(host, usage.start, usage.end).some((l) => l.value === frame.groups.usage), frame.groups.usage);
  const budget = format('host/stats.rs', 'frame budget, the presenting thread, frames {first}..{last}');
  matches(budget, BUDGET_LINE, 'BUDGET_LINE');
  matches(budget, BUDGET_LINE_OTHERS, 'BUDGET_LINE_OTHERS');
  const calls = 'title_is_only_the_title_and_the_calls_are_counted';
  assert.ok(BUDGET_LINE.includes(literalWith(stats, 'the title and its calls together 90.00', { fn: calls })));
  assert.ok(BUDGET_LINE.includes(literalWith(stats, '2000 calls from the title a frame', { fn: calls })));
  const apart = 'the_budget_is_in_the_object_and_other_threads_are_apart';
  assert.ok(BUDGET_LINE_OTHERS.startsWith(literalWith(stats, 'frame budget, the presenting thread, frames 1230..1234', { fn: apart })));
  assert.ok(BUDGET_LINE_OTHERS.endsWith(literalWith(stats, 'not parts of the frame: relay_land', { fn: apart })));
  for (const line of [BUDGET_LINE, BUDGET_LINE_OTHERS]) {
    const parts = /= (.+?) \(sum /.exec(line)[1].split(' + ').map((part) => part.split(' ')[0]);
    for (const name of parts) assert.ok(RUNTIME_PHASES.includes(name), name);
  }
  const queue = source('host/queue_stamps.rs');
  matches(format('host/queue_stamps.rs', 'card queues by GPU timestamps'), GPU_LINE, 'GPU_LINE');
  assert.ok(GPU_LINE.includes(literalWith(queue, 'copy queue per frame: 2.00 fetch lists', { fn: 'a_copy_queue_stamp_appears_in_the_report' })));
});
