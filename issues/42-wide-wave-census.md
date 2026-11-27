# 42 — Name the wide-wave programs the paired form refuses

- **Title:** Name the wide-wave programs the paired form refuses, and fix the largest reason
- **Area:** GPU side: picture, shaders, the card
- **Cards:** nvidia (a device whose waves are 32 lanes; a 64-lane device runs these at their own width)
- **Goal:** On a device whose waves are 32 lanes, each console wave of 64 runs as two host waves (the paired form, on by default). A group of compute programs that need 64-lane waves is still refused there, for reasons nobody has grouped. Produce a per-program census of the group and fix the largest single reason, with a fixture, without touching the programs that already work.
- **What is known, and how:** the warm-up's refusal list (the runtime's own count) on an NVIDIA card: a few dozen refused in all, one group of them wide-wave compute programs the paired form does not take. Suspected for some of them, from their thread-group sizes: groups that span several console waves, for which no 32-lane form exists yet.
- **Where the code is:** `crates/victory-shader/src/emit/paired.rs` (`with_paired_lanes`, `PAIRED_LDS_APART`, `swap_lanes`, and the whole-wave section's `UNKNOWN` and float-sum `ORDER` reasons); `crates/victory-shader/src/emit/mod.rs` (`pairs_for`, `pairs_on`, and the `VICTORY_WAVE_WHY=1` output); `crates/victory-shader/src/emit/host_waves.rs` (how the device's lane counts reach the translator).
- **Files owned:** `crates/victory-shader/src/emit/paired.rs`, `crates/victory-shader/src/emit/paired_tests.rs`, `crates/victory-shader/src/emit/mod.rs`
- **Ruled out:** the programs the paired form already recovers.
- **Done when:** a table in the task comment lists them by program id and reason (the first reason each stops at). A fixture for the largest reason fails before and passes after; `VICTORY_WARP=1 cargo test -p victory-shader --locked paired` passes; with your own copy, the warm-up refuses fewer programs than your own count before the change, by the size of that class, and nothing is newly refused on a 32-lane or a 64-lane device.
- **Test:** a synthetic 64-lane compute program for the chosen reason in `paired_tests.rs`; assert the refusal before and the translation after. Keep `PAIRED_LDS_APART` covered. A test chooses its host with `HostCaps::at_most(32)` (`host_waves.rs`); `VICTORY_HOST_MAX_WAVE=32` does the same for an offline translation, on any machine.
- **Test without the title:** yes, the fixture; the census and the warm-up need your own copy.
- **Out of scope:** the others if their reasons differ; the approximate forms behind `VICTORY_WAVE_PAIRS_APPROX`.
- **Size:** M
- **Needs:** your own copy of the title for the census and the warm-up; the fixture needs none.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
