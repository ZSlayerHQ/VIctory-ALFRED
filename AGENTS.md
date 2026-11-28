# Working on this project with an agent

This repository coordinates work on the project's runtime, a descendant of VIctory (GPL-3.0-or-later), an
open-source runtime that runs a console title's own code on a Windows PC and translates its GPU programs for
Direct3D 12. Every contributor
brings their own copy of the title; nothing from the game is stored here. Work happens in the open: issues are the
job board, claims and run reports are public, and `main` moves only through reviewed pull requests.

## The rule everything else serves

**Claim a task in public before you spend a token on it.** Two people solving the same task is the failure this
repository exists to prevent.

## Start of every session

1. Read [TASKS.md](TASKS.md). It gives the three areas and the files each holds, the work a maintainer holds (take
   none of it), and the claim protocol. Every task is an issue on this repository; its `Area:` line says where it
   sits and its `Cards:` line which graphics cards it concerns.
2. List the open tasks (`gh issue list --state open --label task --limit 200`) and read the ones you could take,
   newest comments first.
3. Read [KNOWN-FAULTS.md](KNOWN-FAULTS.md). A diagnosed fault is settled: link its issue and say what your work adds.
4. Claim one task labelled `open` with one comment, in the form TASKS.md gives:
   `CHECK-IN: <handle> | model: <model name> | effort: <level> | plan: <one line> | branch: <branch>`. Your model
   and effort meet the requirement below.
5. Work only in the files the task lists. If you need a file another task holds, ask its holder on that task.
6. If the task measures anything, run `node harness/tools-check.mjs` and install what it reports missing: the tool
   pack is in [docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md). A speed or picture claim names the established tool that
   measured it.

The three areas, each led by one maintainer:

- **Tools and measurement** (`area: tools`): the harness, the claims bot, this kit, and the runtime's own
  diagnostics and counters.
- **CPU side: speed and memory** (`area: cpu`): the thread that submits each frame, copies and copy-back between
  title memory and the card, page guards and memory tracking, and the console platform layer.
- **GPU side: picture, shaders, the card** (`area: gpu`): the shader translator, textures, descriptors, pipelines,
  draws, and the card's own time.

## Claims, in short

[TASKS.md](TASKS.md) holds the exact comment forms; the claims bot (`.github/workflows/claims.yml`) enforces them.

- **One claim per issue, one claimed issue per agent session.** The first valid `CHECK-IN` holds the issue; a later
  one gets an answer naming the holder. Reason: a claim is only useful if it is unique.
- **Check in at least every 60 minutes**: a new comment starting `PROGRESS:` (what changed, what is next) or
  `CHECK-IN:`, or a push to your claimed branch whose pull request says `Task: #<n>`. After 60 minutes without one
  the claim lapses and anyone may take the issue. Reason: an abandoned agent session otherwise blocks the task
  silently, and an hour is long for an agent.
- **A person working by hand** writes `model: human` and checks in at least every 48 hours, because a person does
  not check in hourly.
- **Release with `RELEASE:`** and what you learned; the issue is free at once. **Finish with a pull request** and
  `CHECK-OUT:`; the claim then holds through review, and the merge is the maintainers' call.
- **You own the task's listed files** until you release it or it lapses. A change needed in another claimed task's
  files goes to its holder as a comment, so two agents never edit one file.
- A lapsed holder's branch stays theirs; build on it if it helps, and credit them.

## Which model

**Model requirement:** work on this project only with GPT-6.2 Sol at high effort or above, Claude Opus 5.5 at
high effort or above, Claude Fable 5.2 at medium effort or above, Gemini 4 Neon at medium effort or above, or a
model that stands at that level on the Artificial Analysis intelligence index.

Reason: this is a highly complex project that needs a model with real understanding of 3D rendering and shaders
from its training, and a pull request from a model below that level costs the maintainers a review that will not
help the project.

- Effort levels rank low, medium, high, xhigh, max; your vendor's own name for a level is fine, in any case.
- The claim's `model:` and `effort:` fields are how the claims bot and the reviewers apply the requirement. The bot
  flags a named model below its effort with the minimum it needs, and a model not in its table as "unknown: a
  maintainer decides"; a flag never takes the claim away, and a maintainer may ask for a rebuild.
- Two rules about how the work is built sit with the quality gates: lock, race and memory-ordering work starts from
  a reviewed design note, and a risky change gets one review from a model of another family.

## Two vendors, one codebase

The runtime is built to run well on NVIDIA and AMD cards, and is tested on Intel too. Where cards differ (wave
width, the descriptor heap's size, ray tracing), the runtime asks the device and takes the path its answer selects,
never a build switch, a vendor check or a separate branch. A pull request names the vendor, card and driver it was
tested on and does not break the others. Reason: one codebase keeps every fix on every card. What the runtime does
per vendor today: [CONTRIBUTING.md](CONTRIBUTING.md), "Two vendors, one codebase".

## Quality gates

Every pull request meets the quality gates in [CONTRIBUTING.md](CONTRIBUTING.md), "Quality gates"; the pull
request template repeats them as a checklist. Among them: the picture is never lowered for speed, a bug fix starts
from a failing fixture, off means off, and tests are not weakened to get a pass. The maintainers hold their own
changes to the same list.

## Hard rules

- **Commit only what may be published.** Assets, parts of the executable, shader dumps, decompiled or disassembled
  code, captures and any bulk extracted table stay in your local logs; names, paths and pointers may be committed.
  Reason: the title is proprietary, and this repository holds only what may be published.
- **Name a shader beside its 16-hex id, and build any names index from your own copy** (`victory shader-names`).
  Reason: a name helps everyone identify the work, and each contributor's index comes from the copy they already
  have, so the repository carries none.
- **Work offline and leave the console's protections alone.** The runtime runs a copy the user already has.
  Reason: defeating encryption is not part of this project and would expose it to takedown.
- **Change no registry keys.** Reason: the runtime must work with a contributor's machine as it is.
- **Write "the title", not its name, and keep that name out of file and folder names.** Reason: the project is
  not affiliated with the title's owners, so nothing it publishes uses the title's name.

## Reporting a run

Paste the whole harness summary into the issue or pull request; `harness/summarize-run.mjs` writes it. It carries
the build, the switches set, the frame and GPU times, the drops and a "what was on screen" section: counts and times,
never pictures. Say which machine produced it: a self-chosen machine id, and the vendor, card and driver (the
runtime's `host adapter:` log line names all three). One run proves little: when a fault is intermittent, say how
many runs showed it, out of how many. A speed or picture claim names the established tool that measured it
([docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md)).
