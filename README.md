# ZSlayer PC Port

<p align="center"><img src="docs/media/alfred-explains.jpg" width="720" alt="Alfred, the project's butler"></p>

One repository for the whole of ZSlayer PC Port: the work of getting a console game (called "the title" here)
running on a Windows PC from your own copy of it. Nothing from the title is stored here.

The public side today is two repositories:
[zslayer-pc-port-runtime](https://github.com/ZSlayerHQ/zslayer-pc-port-runtime) (the runtime) and
[zslayer-pc-port-tools](https://github.com/ZSlayerHQ/zslayer-pc-port-tools) (the test harness and the task list);
this tree is where they come together, with the runtime, the harness and the task list in one place.

## What is here

| Folder | What it is |
|---|---|
| `victory/` | The runtime: a fork of [VIctory](https://github.com/Marvisak/VIctory) by Marvisak (GPL-3.0). It runs the title's own code and translates its graphics for a PC graphics card |
| `harness/` | Scripts that start a test run, record it, measure it and compare runs; live switches for a running session |
| `TASKS.md` and `issues/` | The task list, one page per task, and the check-in protocol |

## Target

The larger console profile only, at least 30 frames a second at 1440p native or 4K, with no lowered render
resolution. Where it stands today is in [KNOWN-FAULTS.md](KNOWN-FAULTS.md) and the open tasks in [TASKS.md](TASKS.md).

## Starting

Read [AGENTS.md](AGENTS.md) (the rules every agent works by) and [CONTRIBUTING.md](CONTRIBUTING.md), then
[TASKS.md](TASKS.md). [victory/README.md](victory/README.md) covers the build and its switches. You need your own
copy of the title, from your own console, on your own disk.

The project is not affiliated with or endorsed by the title's developer or publisher.
