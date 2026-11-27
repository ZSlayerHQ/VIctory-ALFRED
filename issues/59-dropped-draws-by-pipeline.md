# 59 — Name the pipelines whose draws are dropped, in the stats file

- **Title:** Put the dropped draws by pipeline into the live stats file, a second at a time
- **Area:** Tools and measurement
- **Cards:** all
- **Goal:** A flicker (a glass's dirt layer gone for a frame or two, a texture black for a moment) cannot be tied to a pipeline today: the stats give dropped draws only as an average, with no pipeline named. Add to each live stats object the pipelines whose draws were dropped in it (their shaders' 16-hex ids), with counts and the first and last frame, so a drop can be matched to a flicker and then confirmed by skipping that pipeline.
- **What is known, and how:** by reading the code: a dropped draw is noted per reason (`note_drop` and `note_why`, kept in `DROPS` in `crates/victory-gpu/src/pipeline.rs`), and the stats object carries one `dropped` figure per frame (`crates/victory-gpu/src/host/stats.rs`). `VICTORY_SKIP_PIPELINES` already skips pipelines by shader id. Draws the runtime drops never reach Direct3D 12, so no outside tool can see them: this counter is the one place they show.
- **Where the code is:** `crates/victory-gpu/src/pipeline.rs` (`note_drop`, `note_why`, `DROPS`); `crates/victory-gpu/src/draw.rs` (`dropped`); `crates/victory-gpu/src/host/stats.rs` (the stats object and its tests); `VICTORY_LIVE_STATS=<file>` in the runtime's README.
- **Files owned:** `crates/victory-gpu/src/pipeline.rs` (the drop notes only), `crates/victory-gpu/src/host/stats.rs`
- **Ruled out:** a log line per drop (the stats file is read by tools; the log is not a table).
- **Done when:** a test makes synthetic drops for two pipelines across two stats intervals and asserts each object names both with the right counts and frames, at most the ten most-dropped, and nothing when nothing dropped; with no drops the object is byte for byte as today; the GPU crate's tests pass on the software adapter. With your own copy, one run's stats file names the pipelines dropped during the warm-up.
- **Test without the title:** yes, the synthetic test.
- **Judged by:** a run with `VICTORY_SKIP_PIPELINES` set to a named pipeline, in which that pipeline's drops become skips: the counts must agree.
- **Out of scope:** why a pipeline is not ready (pipelines made ahead are held by a maintainer); the harness readers of the stats file.
- **Size:** M
- **Needs:** none for the test; your own copy for the run.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
