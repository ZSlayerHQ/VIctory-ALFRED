# ZSlayer PC Port

<p align="center"><img src="docs/media/alfred-explains.jpg" width="720" alt="Alfred, the project's butler"></p>

<p align="center">
  <b>A console game, running natively on a Windows PC, from your own copy of it.</b><br>
  Open source · built live on stream · nothing from the game is stored here
</p>

---

## What this is

ZSlayer PC Port gets one console game (called **"the title"** everywhere in this repository) running on a Windows
PC with a modern graphics card. It is not an emulator of the whole console and it is not a rewrite of the game:

- **The title's own program runs directly on your processor.** The console and a PC share the same processor
  family, so its code needs no translation.
- **The runtime stands in for the console underneath it:** files, threads, memory, input, sound.
- **The console's graphics programs are translated** into the format Direct3D 12 runs on a PC card.

You bring the title, from your own console, on your own disk. **This repository holds none of it**: no assets, no
executable, no shader dumps, no screenshots, no captures.

> The project is not affiliated with or endorsed by the title's developer or publisher.

## Where it stands

Measured on 2026-11-28 on an NVIDIA RTX 5090, at the larger console's profile and its 4K picture, in the open
world. That day the runtime moved to a faster community line (see [Credits](#credits-and-licence)); the 145 to
155 ms a frame reported here before belonged to the earlier one.

| | Today | Target |
|---|---|---|
| Reaches the open world and stays there | Yes | Yes |
| Frame time (median) | about 40 to 45 ms (22 to 25 frames a second) | 33 ms (30 fps), then 16.7 ms (60 fps) |
| Interiors | lit by a stand-in; the title's own bounce light works behind a switch and costs about 12 ms a frame | the title's own light, at speed |
| Known picture faults | grain on shadow edges, red blocks at the screen's edge, distant characters' heads | none |
| Video memory in use | not yet measured on this base | fits 12 GB cards |
| Ray tracing | the title's own, off by default; switched on it runs as ordinary compute work | on the card's ray-tracing hardware |

The full list of what is wrong and why is in [KNOWN-FAULTS.md](KNOWN-FAULTS.md).

**Left out of this copy:** the base's 1440p output mode, DLSS and frame generation. They work by changing the
title's own code, and files that do that are not published here.

## How it works

```
  the title's own code          runs as it is, on your processor
          │
  platform layer                files, threads, memory, input, sound: what the console provided
          │
  graphics translation          the console's GPU programs become DXIL; its commands become Direct3D 12
          │
  host renderer                 descriptors, memory copies, pipelines, pacing
          │
  your graphics card
```

The hard parts, in the order they cost us time:

1. **Shader translation.** The console's GPU runs its programs 64 threads at a time; NVIDIA cards run 32. Programs
   that share work across all 64 have to be translated so they still agree.
2. **Shared memory.** On the console the processor and the GPU see the same memory. On a PC they do not, so the
   runtime keeps the two sides in step and pays for it in waiting.
3. **Ray tracing.** The title traces its own rays in compute programs. Moving that onto the card's ray-tracing
   hardware is the largest speed-up still open.

## What is in this repository

| Folder | What it is |
|---|---|
| [`victory/`](victory/) | The runtime, in Rust: a descendant of [VIctory](https://github.com/Marvisak/VIctory) (GPL-3.0). It runs the title and translates its graphics |
| [`harness/`](harness/) | The test harness: starts a run, records it, measures it with PresentMon, compares two runs, and summarises the log |
| [`docs/`](docs/) | How we measure ([DIAGNOSTICS.md](docs/DIAGNOSTICS.md)) and the explainer media |
| [`scripts/overlay-art/`](scripts/overlay-art/) | The settings menu's panel art, and the Blender script that makes it |
| [`TASKS.md`](TASKS.md) and [`issues/`](issues/) | The task list, one page per task, and how to claim one |
| [`KNOWN-FAULTS.md`](KNOWN-FAULTS.md) | Every known fault, with what causes it where that is known |
| [`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md) | The rules every contributor works by, person or AI agent |

## Getting started

**You need**

- Windows 11 and a Direct3D 12 graphics card. Tested daily on an RTX 5090; other cards are untested (see below).
- Your own copy of the title, taken from your own console.
- The Rust toolchain named in [`victory/rust-toolchain.toml`](victory/rust-toolchain.toml).

**Then**

1. Read [`victory/README.md`](victory/README.md): it covers the build, the configuration file and every switch.
2. Read [`harness/README.md`](harness/README.md) to start a measured run instead of a bare one.
3. Read [`KNOWN-FAULTS.md`](KNOWN-FAULTS.md) before reporting something as new.

The runtime is offline by design: no online play, no network features.

## How we measure

A number only counts if an established tool produced it. Ours are glue around them, never the judge.

| Question | Tool |
|---|---|
| How long is a frame? | PresentMon |
| Where does the card's time go? | NVIDIA Nsight GPU Trace, PIX |
| Where does a thread's time go, and what is it waiting on? | Windows Performance Recorder, Tracy |
| Did the picture change? | ffmpeg's comparison filters on recorded frames |
| What did the card crash on? | DRED breadcrumbs (Direct3D 12's own crash trail) |

Every result names the card, its video memory and the driver it was measured on. Details are in
[docs/DIAGNOSTICS.md](docs/DIAGNOSTICS.md).

## Graphics cards

| Card | State |
|---|---|
| NVIDIA RTX 5090 (32 GB) | The development card. Everything in this README was measured on it |
| NVIDIA RTX 40 and 30 series | Untested. The shaders translate the same way; **video memory is the open question** on 8 to 16 GB cards |
| AMD Radeon | The runtime's upstream runs on AMD; this line's NVIDIA-specific fixes switch themselves off there. Untested by us |
| Intel Arc | Untested |

If you have one of the untested cards, a run and its results sheet are the most useful thing you can send.
[CONTRIBUTING.md](CONTRIBUTING.md) says how ("Results from another GPU").

## Roadmap

1. **A correct picture.** Fix the grain, the blocks at the screen's edge, and the faults that show when the title's
   own ray-traced light is switched on.
2. **30 frames a second on the development card.** Cut the compute time (ray work onto ray-tracing hardware,
   upscaling from 1080p or 1440p to 4K) and the waiting between the processor and the card.
3. **Smaller cards.** Make memory use follow the card's budget so 12 GB cards can run it.
4. **60 frames a second.**

A lowered render resolution without upscaling, and the smaller console's profile, are not ways to reach the target.

## Contributing

Work is claimed, done and checked in through issues: start at [TASKS.md](TASKS.md).

- **One task, one pull request**, with a test that fails before the change and passes after it.
- **New behaviour sits behind a switch that is off by default**, described where it is read. A fix for a crash or a
  corrupt result may be on by default, and says so.
- **Nothing from the title is ever committed**: no assets, no code listings, no shader dumps, no captures. Names of
  shaders and passes are fine.
- **Never name the title, its developer or its publisher** in code, comments, issues or commit messages.
- **Every risky change is read by two different AI model families** before it merges, and then tested in the game.
- **Results name their card.**

The full rules are in [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md). They apply the same to people
and to AI agents; most of this project is built by agents, in the open, on stream.

## Credits and licence

The runtime descends from [VIctory](https://github.com/Marvisak/VIctory) and carries work from several community
forks of it; who contributed what is recorded in [THIRD-PARTY.md](THIRD-PARTY.md) and in the runtime's own notes.
Everything here is released under the [GNU General Public License v3.0](LICENSE).

Alfred, the butler above, keeps the house in order.
