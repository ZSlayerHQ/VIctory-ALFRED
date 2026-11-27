# 72 — Name the other refused programs

- **Title:** Name the refused programs outside issues 40 to 42, and fix the largest class
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all (say which card's warm-up you read; the 32-lane forms differ from a 64-lane device's)
- **Goal:** Most of the programs the warm-up refuses are covered by issues 40 (sampler branches), 41 (primitive-ordered preambles) and 42 (wide-wave programs). The rest are compute programs that read a descriptor as data, used by the shadow rays, and a few programs refused for one-off lane-mask reasons. List them by program id and reason, then fix the largest class with a fixture.
- **What is known, and how:** the warm-up's refusal list (the runtime's own count) on an NVIDIA card gives them and their reasons; nothing more is known about them.
- **Where the code is:** the reason text of each refusal names where it is raised; the lane-mask reasons are in `crates/victory-shader/src/emit/lower.rs` and `emit/paired.rs`, a descriptor read as data in `emit/lower.rs`. `VICTORY_WAVE_WHY=1` prints why a wide-wave program took its form.
- **Files owned:** to be confirmed by a maintainer once the class is chosen (the files that class's refusal lives in, and a new test file).
- **Ruled out:** the programs of issues 40, 41 and 42.
- **Done when:** a table in the task comment gives them by id, kind and reason; a fixture for the chosen class fails before the change with its reason and passes after; `VICTORY_WARP=1 cargo test -p victory-shader --locked` passes; with your own copy the warm-up refuses fewer by the size of that class and nothing is newly refused.
- **Test without the title:** the fixture; the census needs your own copy.
- **Out of scope:** the ray-tracing structures the shadow-ray programs read.
- **Size:** M
- **Needs:** your own copy of the title for the census.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
