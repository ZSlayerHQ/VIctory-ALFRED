# VIctory-ALFRED: the runtime

This is the program that runs a console game (called "the title" below) on a Windows PC. It runs the title's own
x86-64 code directly on your processor, stands in for the console's system software underneath it, and translates
the title's graphics programs (shaders) so that a PC graphics card can draw them.

It descends from [VIctory](https://github.com/Marvisak/VIctory) by way of two community lines that made it much faster;
[THIRD-PARTY.md](../THIRD-PARTY.md) names them and says what is ours on top. GPL-3.0-or-later, like VIctory.

**It holds no game files.** You need your own copy of the title, from your own console, on your own disk. This
project is not affiliated with or endorsed by the title's developer or publisher.

## Where it stands

Not playable yet. On an NVIDIA RTX 5090, at the larger console's profile and its 4K picture, the open world runs at
about 40 to 45 ms a frame (22 to 25 frames a second), and parts of the picture are still wrong.
[KNOWN-FAULTS.md](../KNOWN-FAULTS.md) says which, and what is known of each cause.

## Build

In this folder:

```powershell
cargo build --profile diag -p victory --features diagnostics --locked
```

The toolchain pinned in `rust-toolchain.toml` installs itself on the first cargo call.

## Configure

Copy `victory.example.toml` to `victory.toml` and set `game_dir` to your own copy of the title.
`cargo run -p victory --locked -- config` shows the resolved configuration and checks it.

## Run

- **Through the harness** (recommended), from the repository root: `harness/run-world.sh --tag first`. It starts the
  runtime with the settings this line is measured with (`harness/settings/noone.env`), records the window and
  writes a summary. `harness/README.md` has the flags.
- **Directly:** `victory run <the title's executable> --execute`, with the same switches in the environment.

The runtime is offline by design: no online play and no network features.

## In the game

- **The top bar** shows the build's name and commit, the frame rate and frame time, and the draws in a frame.
- **F1 opens the settings menu.** It saves only when `VICTORY_MENU_FILE=<file>` is set (the harness's
  `--menu-file`), and what it saved applies at the next start.
- **The menu's panel** is drawn from a `menu-art` folder beside the executable; `harness/snapshot-build.sh` copies
  it there from `scripts/overlay-art/assets/`. Without the folder, or with `VICTORY_MENU_PANEL=0`, the menu is
  plain text.

## Switches

Every switch is an environment variable named `VICTORY_…`, described in the source where it is read. The ones a
first reader meets:

| Switch | What it does |
|---|---|
| `VICTORY_RT_RESOLVE=1` with `VICTORY_ACTIVATE_RAYS=1` | the title's own ray-traced light instead of the stand-in (about 12 ms a frame more) |
| `VICTORY_STRICT_HALVES=1` | refuse the 64-lane programs that have only an approximate form on 32-lane cards |
| `VICTORY_WAVE_WIDE=1` | an exact last-resort form for those programs: slower, and it can hang the card |
| `VICTORY_MENU_PANEL=0` | the settings menu as plain text |

## What this copy leaves out

The base's source files that change the title's own code are not published here, and neither are its scripts that
download other vendors' libraries. Its 1440p output mode, DLSS and frame generation depend on those files, so this
copy does not have them. Small stand-ins that do nothing keep everything else building.

## Tests

`cargo test --workspace --locked`. Tests that draw use the real graphics card unless told otherwise:
`VICTORY_WARP=1` moves the shader crate's tests to the software adapter. A handful of the base's own tests fail or
flake (KNOWN-FAULTS.md, "Tools and tests").

## The rules

Nothing from the title is ever committed: no assets, no executable parts, no shader dumps, no listings, no
captures ([docs/asset-policy.md](docs/asset-policy.md)). Names of shaders and passes are fine. Write "the title",
never its name.

## License

GPL-3.0-or-later (`../LICENSE`), like VIctory and the lines this one descends from.
