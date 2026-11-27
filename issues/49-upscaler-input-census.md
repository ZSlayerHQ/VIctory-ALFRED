# 49 — Census the temporal upscaler's inputs and its card time

- **Title:** Census the title's temporal upscaler: its inputs over two frames and its card time
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all (the census); the later DLSS replacement is NVIDIA's
- **Goal:** The title renders below the output size and upscales to it. Measure on your own copy, and report the sizes you find, across two consecutive frames, what a DLSS replacement needs (the colour, depth and velocity targets, the output, the jitter in pixels, the motion vectors' sign and scale) and how long the upscaler's passes take on the card.
- **What is known, and how:** by reading the pipelines the runtime makes: the upscaling programs all translate. Not known: which stage a replacement would stand in for, the passes' card time (a few ms leaves this for later; tens of ms makes it the place DLSS would go), the jitter and the motion-vector convention.
- **Where the code is:** `crates/victory-gpu/src/capture.rs` (`note_target`, `dump_targets`, the target list); `crates/victory-gpu/src/passes.rs` (`label_hash`, `By`, `PassName`); `crates/victory-gpu/src/draw.rs` (`bind`, where a pass begins); `crates/victory-gpu/src/events.rs` names the runtime's own work for outside tools.
- **Files owned:** `crates/victory-gpu/src/capture.rs`, `crates/victory-gpu/src/passes.rs`; ask before editing `draw.rs`
- **Ruled out:** nothing yet.
- **Done when:** with your own copy, two consecutive frames give one census line per frame naming the colour, depth and velocity targets by format and size, the output, the jitter x and y in pixels, and the velocity target's first non-zero values with their sign and scale; the comment gives the upscaler's passes' card time from PIX or Nsight, and says which figures are certain and which inferred. The census record has a unit test on made-up values; the GPU crate's tests pass on the software adapter.
- **Test without the title:** the census record's unit test; the census itself needs your own copy.
- **Judged by:** PIX or Nsight for the passes' card time.
- **Out of scope:** writing the DLSS evaluate; changing the upscaler; judging image quality.
- **Size:** M
- **Needs:** your own copy of the title; PIX or Nsight.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
