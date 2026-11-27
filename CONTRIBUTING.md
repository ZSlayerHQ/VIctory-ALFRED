# Contributing and claiming work

A task moves in one direction: claim, branch, gate, review, merge. The claim is what keeps two people out of the
same files; the quality gates and the review are what keep `main` sound. The full protocol, with the exact
comments to post, is in [TASKS.md](TASKS.md).

## Claiming, checking in and releasing

[TASKS.md](TASKS.md) is the protocol, and the claims bot enforces it: how to claim an issue, how often to check in,
how a claim lapses, how to release a task or finish it, and who owns which files meanwhile.

**Model requirement:** work on this project only with GPT-6.2 Sol at high effort or above, Claude Opus 5.5 at
high effort or above, Claude Fable 5.2 at medium effort or above, Gemini 4 Neon at medium effort or above, or a
model that stands at that level on the Artificial Analysis intelligence index. The reason, and how the claims bot
applies it, are in [AGENTS.md](AGENTS.md), "Which model".

## One area, one task per pull request

Each issue names one of the three areas ([TASKS.md](TASKS.md), "The three areas") and the files it may change.
Runtime paths are under `victory/`. Cargo is run inside `victory/` (there is no `Cargo.toml` at the repository
root), and the pinned toolchain is fetched on the first cargo call. An issue whose files are not yet certain says
"to be confirmed by a maintainer"; ask on the task before you edit outside what it lists. Keep a pull request to one
task even when two changes are ready: split it.

## Two vendors, one codebase

The runtime is built to run well on both NVIDIA and AMD graphics cards, and is tested on Intel too. Where the
cards differ, the runtime asks the device when it starts and takes the path the answer selects. There is no build
feature, launch option or branch per vendor (the build features are `diagnostics` and the translator's `host`
and `adapter`, none of them per vendor), and a change adds none. Reason: one codebase keeps every fix on every card.

What the runtime does today, where the cards differ:

- **Wave width.** The console's programs run in waves of 64 lanes; AMD cards have 64-lane waves, NVIDIA cards 32.
  The host reads the device's own lane counts (`WaveLaneCountMin` and `WaveLaneCountMax`, in
  `crates/victory-gpu/src/host.rs`) and the translator chooses each program's form from them
  (`crates/victory-shader/src/emit/host_waves.rs`, `emit/mod.rs`): a device with 64-lane waves runs a console
  wave at its own width; on a device whose waves are 32 lanes a console wave runs as two host waves (the paired
  form, on by default; `VICTORY_WAVE_PAIRS=0` turns it off) or in one of the other 32-lane forms; a program no form
  fits is refused when translated, with its reason. `VICTORY_HOST_MAX_WAVE` names a host for an offline
  translation.
- **Descriptor heap size.** The host tries a shader-visible view heap of 2^21 views and takes 1,000,000 where the
  device refuses it, as NVIDIA's does (`probe_view_limit`, `crates/victory-gpu/src/descriptor/host.rs`). Every
  device clamps view indices to a null slot at the heap's end. A map from the title's slots to the host's
  (`VICTORY_SLOT_MAP=1`, off by default) is the path for the smaller heap.
- **Ray tracing.** The title's ray-tracing structures are not built on any card yet; when they are, the path is
  chosen from the tier the device reports.
- **Instruction results.** Where drivers differ in an operation's result (a float-to-half rounding, an unordered
  compare), the translator writes the exact form for every card instead of relying on the driver's.
- **Which card.** The host takes the hardware adapter Windows ranks first for high performance that supports
  shader model 6.6, logs it as `host adapter: <name> (vendor 0x…, device 0x…, driver …)`, and
  `VICTORY_ADAPTER` overrides the choice.

A pull request says on which vendor, card and driver it was tested, and that it does not break the others. An
issue marked for one vendor still needs that statement for the rest: the software adapter's tests count where the
behaviour is not vendor-specific; where it is, say which vendors you could not test, so a maintainer can.

## Quality gates

Every pull request meets each gate below; the pull request template repeats them as a checklist, and a reviewer
checks each one. Run the cargo commands in `victory/`.

- **The picture is never lowered for speed.** The target is the larger console's profile (`--console series-x`)
  at 1440p native or 4K, or upscaled to 4K from 1080p or 1440p; the smaller profile and a lowered render
  resolution are not ways to reach a frame-rate goal. Reason: a frame time bought by drawing less is not progress
  toward the goal.
- **A bug fix starts from a failing fixture.** Write the smallest test that shows the bug (for the translator, a
  synthetic program in the style of `crates/victory-shader/src/emit/wave64_tests.rs`), see it fail, then fix it and
  commit both together. A port of working code or a feature behind an off switch may get its test afterwards.
  Reason: a bug without a fixture comes back.
- **Off means off.** With its switch off, a feature leaves the default path unchanged, byte for byte; say how you
  checked. Reason: a switch that changes the default path is an untested change to everyone's build.
- **A loop cap is a diagnostic.** `VICTORY_SHADER_LOOP_LIMIT` finds a loop that runs too long; raising it is never
  the fix for a hang or a slow frame. Reason: a cap hides the loop instead of bounding it.
- **Tests are not weakened.** Do not remove or weaken a test to get a pass. When a task, a test or its premise
  looks wrong, say so on the issue instead of working around it. Reason: a pass bought by changing the test hides
  the defect it was written to catch.
- **The commands pass.** `cargo clippy --workspace --all-targets --locked -- -D warnings`, the same with
  `--all-features`, `cargo test --workspace --locked`, the same with `--all-features`, and `cargo xtask policy`;
  for a harness change, `node --test "harness/**/*.test.mjs"` and `bash -n` on each changed script. Reason: these
  are the runtime's CI checks and the tests that exist only behind a feature, and review time is for what a command
  cannot catch.
- **GPU tests run on the software adapter while you work.** The GPU crate:
  `VICTORY_WARP=1 VICTORY_D3D12_WARP=1 cargo test -p victory-gpu --features diagnostics --locked` (it needs both
  variables and the feature); the translator: `VICTORY_WARP=1 cargo test -p victory-shader --locked`. Run the full
  commands above on your card once the change is ready. Reason: a test that stalls a queue on a real card resets
  its driver and blanks every display on it.
- **A fixture that can stall a queue refuses a card.** Such a test runs only on the software adapter and refuses
  a hardware one, as `production_child` in `crates/victory-gpu/src/host/tests/fetch.rs` does. Reason: on a card,
  each run of it resets the card's engine.
- **A measured claim names its tool.** A speed or picture claim gives the figures before and after and the
  established tool that measured them: PresentMon for frame times, a system trace (WPA) or Tracy for a thread's
  time and lock waits, PIX or Nsight for the card's time per pass, ffmpeg's filters for picture differences
  ([docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md)). The runtime's own counters may explain a figure, not stand in for
  it. Reason: those tools have years of use behind them, and three of this project's own measuring tools
  misreported within a day of being written.
- **Tested on a named card, and the others still pass.** The pull request names the vendor, card and driver it
  was tested on and says how the other vendors are covered ("Two vendors, one codebase", above). Reason: a fix for
  one card that breaks another is a regression.
- **A risky change gets one review from another model family.** Shader translation, the memory model, and
  anything on by default that changes the picture are reviewed once before merge by a model of a different family
  from the one that wrote the change; the pull request names it. Reason: another family catches what this one
  misses, where a bug is costly.
- **Lock, race and memory-ordering work starts from a design note.** When correctness rests on lock order, a race
  or memory ordering, post a short design note on the task and have it reviewed before you write code. Reason:
  such faults rarely show in a test run and cost the most to find later.
- **Nothing from the title is committed.** No assets, executable bytes, shader dumps, decompiled or disassembled
  code, captures or bulk extracted tables ([the asset policy](victory/docs/asset-policy.md)); names, paths and
  pointers are fine. Reason: the title is proprietary, and this repository holds only what may be published.

The maintainers hold their own changes to this same list.

## Results from another GPU

A run is evidence only if it can be compared with someone else's.

- Paste the whole `summarize-run.mjs` summary. It already carries the build, the switches set and "what was on screen".
- Add a `harness machine:` line at the top of your log paste: a self-chosen machine id, the GPU vendor, card and
  driver version, RAM, and the Windows build. The same id on later runs is what makes them comparable.
- Run with the switches the issue names. Say whether the cache was cold or warm.
- For an intermittent fault, report how many runs showed it out of how many, and what differed.
- Captures stay on your machine. Describe what you saw; post counts and times.

## What a reviewer checks

- The issue's "done when" is met, and every quality gate above holds.
- The diff stays within the task's listed files.
- The claim's model and effort meet the model requirement (above), or a maintainer has accepted them.
- Run evidence, where the issue asks for it: the summary whole, with the machine line.
- The commits say what changed and why, in one or two paragraphs. Formatting-only churn is split out.
