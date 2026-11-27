#!/usr/bin/env node
// Condense a VIctory run log into one page.
// usage: node summarize-run.mjs <log> [--json]
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import nodePath from 'node:path';
import { parseExtras, parseLogLines, readLogLines, runStarts } from './parse-run.mjs';
import { aggregate, collect } from './run-lines.mjs';
import { presentMonCsvForLog, readPresentMon, capturesRoot } from './presentmon.mjs';
import { annotate, defaultNamesFile, loadNames } from './name-hashes.mjs';
import { loadLocalEnv } from './local-env.mjs';
import { pictureLines } from './summary-picture.mjs';
import { dropWebAddresses, hasWebAddress, withheldText } from './web-address.mjs';
import { analyseVideo, cropVerdicts, loadCrops, loadReference } from './picture-check.mjs';
import {
  clock,
  formatStretch,
  frameSamples,
  joinLog,
  recordingStart,
  sampleLuma,
  timeline,
  totals,
} from './screen-timeline.mjs';

// This machine's harness paths (the captures root, where the reference's crops and numbers live).
loadLocalEnv();

// The run's own screen recording and contact sheets, from the capture line run-world.sh writes.
function screenRecord(lines) {
  const capture = lines.map((line) => line.match(/^capture: (.+)$/)).find(Boolean);
  if (!capture) return { file: null, frames: 0, sheets: 0 };
  const file = nodePath.join(capture[1].trim(), 'screen.mp4');
  if (!fs.existsSync(file)) return { file: null, frames: 0, sheets: 0 };
  const sheetsDir = nodePath.join(nodePath.dirname(file), 'sheets');
  const sheets = fs.existsSync(sheetsDir)
    ? fs.readdirSync(sheetsDir).filter((name) => /^sheet-\d+\.jpg$/i.test(name)).length
    : 0;
  const probe = spawnSync(
    'ffprobe',
    ['-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'default=nw=1:nk=1', file],
    { encoding: 'utf8' },
  );
  const frames = Number((probe.stdout || '').trim());
  return { file, frames: Number.isFinite(frames) && frames > 0 ? frames : null, sheets };
}

// The picture block: measured numbers from the recording against the reference beside the crops file under the
// captures root's reference folder. A missing reference is said plainly and the recording is left undecoded.
async function pictureForRun(record) {
  if (!record.file) return null;
  const directory = nodePath.join(capturesRoot(), 'reference');
  try {
    const reference = loadReference(nodePath.join(directory, 'reference.json'));
    if (!reference) {
      return { reference: false, frames: record.frames, verdict: 'no-reference', summary: 'picture: no reference' };
    }
    const crops = loadCrops(nodePath.join(directory, 'crops.json'));
    const result = await analyseVideo(record.file, { crops, reference, stepSeconds: 2, maxWidth: 1280 });
    return {
      verdict: result.verdict,
      reference: true,
      frames: result.frames.length,
      frozen: result.frozen,
      crops: cropVerdicts(result.frames),
      worstFrame: result.worstFrame,
      summary: result.summary,
    };
  } catch (error) {
    return { reference: true, frames: null, verdict: 'unavailable', summary: `picture: unavailable (${error.code || 'failed'})` };
  }
}

const path = process.argv[2];
if (!path) {
  console.error('usage: node summarize-run.mjs <log> [--json]');
  process.exit(2);
}
const asJson = process.argv.includes('--json');

// A log that holds two runs would add them together (the validator's joined log of 2026-11-26: 7675 s, counts
// doubled): only the last run is summarized, and the page says so.
const allLines = readLogLines(path);
const starts = runStarts(allLines);
const lines = starts.length ? allLines.slice(starts.at(-1).index) : allLines;
const { summary: out, details, firstTs, lastTs } = parseLogLines(lines, path);
const extras = parseExtras(lines, firstTs, lastTs);
out.ending = extras.ending;
out.runsInLog = starts.length
  ? { count: starts.length + 1, summarized: 'last', fromLine: starts.at(-1).index + 1, reason: starts.at(-1).reason }
  : { count: 1 };
// Warn mode's tally. A run whose log carries no watchdog stall line has no field, and no line below.
if (extras.watchdog) out.watchdog = extras.watchdog;
const presentmon = readPresentMon(presentMonCsvForLog(lines, path));
if (presentmon) out.presentmon = presentmon;

const duration = out.durationSeconds;
const last = out.lastFrameLine;
const worst = out.worstFrames;
const refused = out.refusedPipelines.hashes;
const refusedCount = out.refusedPipelines.count;
const translated = out.translated;
const slow = out.slowSubmissions;
const missingBy = out.missingTextures;
const driverEvents = out.driverEvents;
const failures = out.failures;
const { slowLists, slowCmds, hangs } = details;

// The speed work's lines (capabilities, thread clock, table constants, buffer copies, settle reasons, pipelines left
// out), over the same part of the run compare-runs measures: after the settle line, when there is one.
const settledAt = lines.findIndex((line) => /harness: settled at/.test(line));
const speed = aggregate(collect(lines, settledAt >= 0 ? settledAt + 1 : 0));
out.speedLines = speed;

// The recording's own numbers. The picture block rides in the JSON as well as the readable page.
const record = screenRecord(lines);
const picture = await pictureForRun(record);
if (picture) out.picture = picture;

// No line holding a web address leaves this script (web-address.mjs): in the JSON such a string is taken out and
// counted in `withheldLines`; on the page the line is not printed and the count is said at the end.
if (asJson) {
  const safe = dropWebAddresses(out);
  console.log(JSON.stringify({ ...safe.value, withheldLines: safe.withheld }, null, 2));
  process.exit(0);
}
let withheldLines = 0;
{
  const write = console.log;
  console.log = (...args) => {
    const text = args.map((arg) => (typeof arg === 'string' ? arg : String(arg))).join(' ');
    const kept = text.split('\n').filter((line) => {
      if (!hasWebAddress(line)) return true;
      withheldLines += 1;
      return false;
    });
    if (kept.length) write(kept.join('\n'));
  };
}

// When a shader-name table exists, every hash in the readable output gains its shader's name. JSON keeps its
// exact hash strings (it exits above).
const names = loadNames(defaultNamesFile());
if (names.size) {
  const write = console.log;
  console.log = (...args) =>
    write(...args.map((arg) => (typeof arg === 'string' ? annotate(arg, names) : arg)));
}

// A field absent from the log prints n/a: never zero, never a dash that reads as a blank.
const NA = 'n/a';
const f = (x, d = 1) => (x === null || x === undefined || !Number.isFinite(Number(x)) ? NA : Number(x).toFixed(d));
console.log(`# Run summary — ${path}`);
console.log(`comparable: ${out.comparable}`);
console.log(`observers on: ${out.observers ? out.observers.join(', ') : 'not in the log'}`);
if (out.switches.length) {
  console.log(
    `switches set: ${out.switches
      .map((entry) => `${entry.name}=${entry.value} (${entry.on ? 'on' : 'off'})`)
      .join(', ')}`,
  );
}
// PresentMon is the judge of frame times (the owner, 2026-11-26): its figures come first, before any of the
// runtime's own, and its absence is said, never silent.
if (!presentmon) {
  console.log('PresentMon: not run');
} else if (!presentmon.frames) {
  console.log('PresentMon: ran, but its CSV holds no frame times');
} else {
  console.log(
    `PresentMon: ${presentmon.frames} frames — median ${f(presentmon.medianMs)} ms, p95 ${f(presentmon.p95Ms)} ms, GPU busy median ${f(presentmon.gpuMedianMs)} ms`,
  );
}
if (starts.length) {
  console.log(
    `warning: this log holds ${starts.length + 1} runs (${starts.at(-1).reason} at line ${starts.at(-1).index + 1}); this summary covers the last one only`,
  );
}
if (out.unreadFrameLines) {
  console.log(
    `warning: ${out.unreadFrameLines} frame line${out.unreadFrameLines === 1 ? '' : 's'} (lines with ms/frame) ${out.unreadFrameLines === 1 ? 'is' : 'are'} in a form this parser does not read; their figures are left out`,
  );
}
console.log(
  `Range: ${duration === null ? NA : duration + ' s'}; last presented frame: ${out.lastFrame && out.lastFrame.frame !== null ? out.lastFrame.frame : NA}`,
);
console.log(`Ending: ${out.ending.text}`);
if (out.watchdog) {
  console.log(`watchdog: ${out.watchdog.stalls} stalls, longest ${out.watchdog.longestSeconds} s`);
}
console.log(
  `Screen record: ${
    record.file
      ? `${record.file} (${record.frames === null ? '?' : record.frames} frames, ${record.sheets} sheets)`
      : 'none'
  }`,
);
for (const line of pictureLines(picture)) console.log(line);
console.log(
  `FPS: ${out.fps.samples} samples, median ${f(out.fps.median)}, min ${f(out.fps.min)}, max ${f(out.fps.max)}, last ${f(out.fps.last)}`,
);
// Which frame-line shape was read: from job/timing-truth on each line's figures are a mean over every frame since the
// line before (its window clause says which frames, how many and over how long); an older line's were a sample. The
// windows joined are weighted by their frames, which a median of the lines is not.
const SHAPE_WORDS = {
  window: 'the window shape (each a mean over every frame since the line before)',
  older: 'the older shape (a sample, no window clause)',
};
const shapes = out.frameLines;
if (shapes.window + shapes.older) {
  console.log(
    `Frame lines: ${shapes.window + shapes.older}, ${
      shapes.window && shapes.older
        ? `mixed: ${shapes.window} in ${SHAPE_WORDS.window}, ${shapes.older} in ${SHAPE_WORDS.older}`
        : `all in ${shapes.window ? SHAPE_WORDS.window : SHAPE_WORDS.older}`
    }`,
  );
}
const fw = out.frameWindow;
console.log(
  fw
    ? `Frame windows: frames ${fw.firstFrame}..${fw.lastFrame}, ${fw.frames} frames in ${f(fw.wallMs, 0)} ms: ${f(fw.meanMs)} ms/frame (${f(fw.fps)} fps), weighted by frames; the last line's p50 ${f(fw.lastP50Ms)}, p95 ${f(fw.lastP95Ms)} ms`
    : 'Frame windows: n/a (no frame line has the window clause)',
);
if (!out.budget) console.log('Frame budget: n/a (no frame budget line: a build without diagnostics, or from before job/timing-truth)');
if (out.costs) {
  console.log(
    `Last frame: ${last.frame === null ? '' : `frame ${last.frame}, `}${f(last.ms)} ms — preparing ${f(last.preparing)} (copies ${f(last.copiesMs)} ms / ${f(last.copiesMb)} MB, textures hashed ${f(last.texturesMs)} ms / ${f(last.texturesMb)} MB), host waits ${f(last.hostWaitsMs)} ms, readback ${f(last.backMs)} ms / ${f(last.backMb)} MB; draws ${last.draws}, dropped ${last.dropped} (${last.making} being made), indirect ${last.indirect}`,
  );
}
if (out.copyStats) {
  const c = out.copyStats;
  const parts = [];
  if (c.descriptorCopySyncMs !== null) {
    parts.push(
      `descriptor-copy sync ${f(c.descriptorCopySyncMs, 2)} ms (${c.descriptorCopyRecords === null ? NA : c.descriptorCopyRecords} records)`,
    );
  }
  if (c.askedAboutMb !== null) parts.push(`${f(c.askedAboutMb)} MB asked about`);
  if (c.distinctPages !== null)
    parts.push(`${c.distinctPages} distinct pages since the last present`);
  if (c.guardCount !== null) parts.push(`${c.guardCount} guards in ${f(c.guardsMs)} ms`);
  else if (c.guardsMs !== null) parts.push(`guards ${f(c.guardsMs)} ms`);
  if (parts.length) console.log(`Copy stats: ${parts.join(', ')}`);
}
// A run under a capture tool: its frame is not comparable, and its exit code is the tool's.
if (lines.some((line) => /^harness: captured by RenderDoc/.test(line))) {
  console.log("Captured: RenderDoc \u2014 the frame is not comparable, and the exit code is RenderDoc's.");
}
// A run under the GPU profiler: its frame times are the profiler's, and its exit code is Nsight's.
if (lines.some((line) => /^harness: profiled by Nsight GPU Trace/.test(line))) {
  console.log("Profiled: Nsight GPU Trace — frame times are not comparable, and the exit code is the profiler's.");
}
// The settle, when the run used the settled clock: when the scene was ready, or that it never was.
if (extras.settled) {
  console.log(
    `Scene settled: ${extras.settled.seconds} s after the first heavy line (waiting ${extras.settled.waiting ?? NA}, ${extras.settled.fps} fps).`,
  );
} else if (extras.settledTimeout) {
  console.log('Scene never settled: the run hit --settled-timeout.');
}
// The picture first (G-017): what the recording shows, stretch by stretch, with the log's figures for the same
// wall-clock time. A heavy scene that is black can no longer pass as "world reached".
if (record.file) {
  console.log('\n## What was on screen');
  try {
    const samples = sampleLuma(record.file);
    const stretches = joinLog(timeline(samples), { start: recordingStart(lines), frames: frameSamples(lines) });
    const all = totals(stretches);
    const black = all.longestBlack
      ? `; longest black ${clock(all.longestBlack.start)}-${clock(all.longestBlack.end)}`
      : '';
    console.log(
      `- ${samples.length} s: black ${all.seconds.black} s, near-black ${all.seconds.nearBlack} s, picture ${all.seconds.picture} s${black}`,
    );
    for (const stretch of stretches) console.log(`- ${formatStretch(stretch)}`);
  } catch (error) {
    console.log(`- unavailable: ${error.message}`);
  }
}
// The presenting thread's frame budget: its phases add up to the frame (unaccounted is our queue calls' time in no
// named phase), at the line's own printed precision. The identity is checked on every line within rounding (parse-run
// budgetIdentity: half a unit of the last printed place per figure added); a line outside it is a warning.
if (out.budget) {
  const b = out.budget;
  const fx = (x) => f(x, b.decimals);
  const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
  console.log('\n## Frame budget (the presenting thread)');
  console.log(
    `- ${plural(b.lines, 'line')}, frames ${b.firstFrame}..${b.lastFrame} (${b.frames} frames)${b.lines > 1 ? ', means weighted by frames' : ''}: ` +
      `${fx(b.frameMs)} ms/frame = ${Object.entries(b.phases).map(([name, ms]) => `${name} ${fx(ms)}`).join(' + ')} (sum ${fx(b.sumMs)})`,
  );
  const id = b.identity;
  if (id.failedLines) {
    const w = id.worst;
    console.log(
      `warning: the frame budget's phases do not sum to the frame on ${id.failedLines} of ${plural(id.checkedLines, 'line')} ` +
        `(worst at frames ${w.firstFrame}..${w.lastFrame}: phases ${fx(w.phaseSumMs)} ms against sum ${fx(w.sumMs)} and frame ` +
        `${fx(w.frameMs)} ms; tolerance ${w.toleranceMs.toFixed(3)} ms)`,
    );
  } else {
    console.log(
      `- identity: the phases sum to the frame on all ${plural(id.checkedLines, 'line')} within rounding (tolerance: half a unit ` +
        `of the last printed place per figure added, ${id.toleranceMs.toFixed(3)} ms at most here)`,
    );
  }
  if (id.unknownPhases?.length) {
    console.log(
      `warning: the frame budget names ${plural(id.unknownPhases.length, 'phase')} the runtime's list does not (${id.unknownPhases.join(', ')}): ` +
        'its line shape has moved, so its figures are not read as the runtime means them',
    );
  }
  // The title and its calls into the runtime have no processor time one by one: only all of them together.
  const cpu = [`the title and its calls together ${fx(b.titleAndCallsCpuMs)}`, ...Object.entries(b.cpu).map(([name, ms]) => `${name} ${fx(ms)}`)];
  console.log(
    `- on the processor: ${cpu.join(', ')}` +
      `${b.cpuMissingLines ? `; the counter's rate was unknown for part of ${plural(b.cpuMissingLines, 'line')}: left out there` : ''}`,
  );
  console.log(
    `- ${f(b.scopes, 0)} scopes a frame, ${f(b.calls, 0)} calls from the title a frame, the largest stretch of title between two ${fx(b.largestTitleGapMs)} ms`,
  );
  const others = Object.entries(b.others);
  console.log(
    `- other threads per frame, overlapping, not parts of the frame: ${
      others.length ? others.map(([name, o]) => `${name} ${fx(o.ms)} (cpu ${fx(o.cpuMs)})`).join(', ') : 'nothing measured'
    }`,
  );
}
console.log('\n## Worst frames');
for (const w of worst) {
  console.log(
    `- ${w.ts} ${f(w.fps)} fps / ${f(w.ms)} ms: copies ${f(w.copiesMs)} ms ${f(w.copiesMb)} MB, host waits ${f(w.hostWaitsMs)} ms, draws ${w.draws}, dropped ${w.dropped}`,
  );
}
console.log('\n## Pipeline health');
console.log(`- refused: ${refusedCount}${refusedCount ? ' — ' + refused.slice(0, 8).join('; ') : ''}`);
console.log(
  `- translated this run: ${translated.total}${Object.keys(translated.byStage).length ? ' (' + Object.entries(translated.byStage).map(([k, v]) => `${k} ${v}`).join(', ') + ')' : ''}`,
);
if (slow.length) {
  const top = [...slow].sort((a, b) => b.ms - a.ms);
  console.log(`- slow submissions: ${slow.length} over 250 ms; worst:`);
  for (const s of top.slice(0, 8))
    console.log(`  - ${s.ms} ms, ${s.lists} lists, ${s.entries} pipelines`);
  if (top.length > 8) console.log(`  - … and ${top.length - 8} more`);
}
if (slowLists.length) {
  const top = [...slowLists].sort((a, b) => b.ms - a.ms).slice(0, 5);
  console.log('- slow host lists:');
  for (const s of top) console.log(`  - ${s.ms} ms, ${s.commands} commands`);
}
if (slowCmds.length) {
  const top = [...slowCmds].sort((a, b) => b.ms - a.ms).slice(0, 8);
  console.log('- slowest commands:');
  for (const c of top) console.log(`  - ${c.ms} ms: ${c.desc.slice(0, 150)}`);
}
if (Object.keys(speed.leftOut).length) {
  console.log('- left out:');
  for (const [reason, perFrame] of Object.entries(speed.leftOut).sort((a, b) => b[1] - a[1])) {
    console.log(`  - ${f(perFrame)} a frame: ${reason}`);
  }
}
// What the speed work's lines say per frame: the card's capabilities, the two counter lines, and every settle
// reason with its count and its milliseconds (not just the top three).
const settleReasons = Object.entries(speed.settleReasons).sort((a, b) => (b[1].count ?? 0) - (a[1].count ?? 0));
const hasSpeed =
  speed.capabilities ||
  speed.tableConstants.inPlace !== null ||
  speed.bufferCopies.onCard !== null ||
  settleReasons.length ||
  Object.keys(speed.threads.named).length;
if (hasSpeed) console.log('\n## Card and CPU per frame');
if (speed.capabilities) console.log(`- host capabilities: ${speed.capabilities.raw}`);
if (speed.tableConstants.inPlace !== null) {
  console.log(`- table constants: ${f(speed.tableConstants.inPlace)} read in place, ${f(speed.tableConstants.copied)} copied`);
}
if (speed.bufferCopies.onCard !== null) {
  console.log(
    `- buffer copies: ${f(speed.bufferCopies.onCard)} on the card (${f(speed.bufferCopies.onCardMb)} MB), ` +
      `${f(speed.bufferCopies.onCpu)} on the CPU (${f(speed.bufferCopies.onCpuMb)} MB), ` +
      `${f(speed.bufferCopies.settled)} settled flights (${f(speed.bufferCopies.settledMb)} MB)`,
  );
}
if (settleReasons.length) {
  console.log('- settle reasons:');
  for (const [reason, entry] of settleReasons) {
    const parts = [];
    if (entry.count !== null) parts.push(`${f(entry.count)} a frame`);
    if (entry.ms !== null) parts.push(`${f(entry.ms)} ms`);
    console.log(`  - ${reason}: ${parts.join(', ') || NA}`);
    for (const [sub, n] of Object.entries(entry.by)) console.log(`    - by ${sub}: ${f(n)}`);
  }
}
// The two instrument lines: a view left null counted while it is written, and the first-touch scoreboard. A run
// whose first-touch line said NOT CLEAN says so here, in those words.
if (speed.nullMagenta || speed.firstTouch) console.log('\n## Run health');
if (speed.nullMagenta) {
  const nm = speed.nullMagenta;
  console.log(
    `- null magenta: ${f(nm.substituted)} substituted a frame (worst ${f(nm.worstSubstituted)}), ` +
      `${f(nm.neverMade)} never made, ${f(nm.waiting)} waiting, ${f(nm.cannotMake)} cannot make, ` +
      `${f(nm.unmapped)} unmapped, ${f(nm.stale)} stale`,
  );
  if (nm.examples.length) console.log(`  - examples: ${nm.examples.slice(0, 6).join(', ')}`);
}
if (speed.firstTouch) {
  const ft = speed.firstTouch;
  console.log(
    ft.clean
      ? `- first touch: clean (${f(ft.guardsPutOn)} guards put on, ${f(ft.faultsTaken)} faults taken), cost ${ft.costMs.toFixed(3)} ms a frame`
      : `- first touch: **NOT CLEAN** — ${f(ft.worstSilent)} guards went with no fault (${f(ft.guardsPutOn)} put on, ${f(ft.faultsTaken)} faults taken), cost ${ft.costMs.toFixed(3)} ms a frame`,
  );
}

const threads = Object.entries(speed.threads.named)
  .sort((a, b) => b[1].user + b[1].kernel - (a[1].user + a[1].kernel))
  .slice(0, 12);
if (threads.length) {
  console.log('\n## Threads');
  for (const [name, entry] of threads) console.log(`- ${name}: ${f(entry.user + entry.kernel)} ms (${f(entry.pct)}%)`);
  const total = speed.threads.total;
  if (total) console.log(`- total: ${f(total.user + total.kernel)} ms (${f(total.pct)}%)`);
}
if (Object.keys(missingBy).length) {
  console.log('\n## Textures with no host copy');
  for (const [k, v] of Object.entries(missingBy)) console.log(`- ${k} x${v}`);
}
if (hangs.length) {
  console.log('\n## Hung commands');
  for (const h of hangs) {
    const pipeline = h.pipelineUnknown
      ? `pipeline unknown (${h.pipelineUnknown})`
      : h.command;
    console.log(`- stopped at command ${h.at}/${h.of} (op ${h.op}): ${pipeline}`);
  }
}
console.log('\n## Driver events');
if (!driverEvents.total) {
  console.log('- none');
} else {
  console.log(`- MMU Fault: ${driverEvents.counts.mmuFault}`);
  console.log(`- Multiple Warp Errors: ${driverEvents.counts.multipleWarpErrors}`);
  console.log(`- other id 13 texts: ${driverEvents.counts.otherId13}`);
  console.log(`- id 153: ${driverEvents.counts.id153}`);
  for (const set of driverEvents.esrValueSets)
    console.log(`- ESR values: ${set.values.join(' ')} x${set.count}`);
}
if (failures.length) {
  console.log('\n## Failures');
  for (const x of failures) console.log(`- ${x}${x.length >= 300 ? '…' : ''}`);
}
if (withheldLines) console.log(`\n- ${withheldText(withheldLines)}`);
