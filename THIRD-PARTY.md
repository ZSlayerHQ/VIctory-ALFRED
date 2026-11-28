# Third-party code and credits

The runtime in `victory/` is other people's work with ours on top. One line per project whose code is in this
repository. Each project's own repository lists its contributors; they are not repeated here.

| Project | What is here | Licence |
|---|---|---|
| [VIctory](https://github.com/Marvisak/VIctory) | the runtime itself: the loader, the console platform, the Direct3D 12 host and the shader translator | GPL-3.0-or-later |
| [readmet3xt-lab/foxhound](https://github.com/readmet3xt-lab/foxhound) | the community line the base builds on: its work on submission, memory copies and translation | GPL-3.0-or-later |
| [trustno1-coder/VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA) | the base this repository's runtime starts from since 2026-11-28, the fastest line we measured, with its settings menu | GPL-3.0-or-later |

**Ours on top:** the top bar and its timing, the plain-language names and the panel of the settings menu (art and
the script that makes it are in `scripts/overlay-art/`), translator fixes with their fixtures, and the test harness
in `harness/`.

**Left out of this copy on purpose:** the base's source files that change the title's own code (its 1440p output
mode, DLSS and frame generation depend on them) and its scripts that download other vendors' libraries. Small
stand-ins that do nothing keep the rest building.

Before 2026-11-28 this repository carried an earlier runtime line; it remains in this repository's history.
