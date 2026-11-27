# 60 — A pipeline drawn into a target of another format

- **Title:** Find whether a pipeline drawn into a render target of another format makes an invalid draw, and fix it if so
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** A host pipeline is made with the render-target and depth formats the title's pipeline declares. If the title then draws with targets of other formats bound, Direct3D 12 calls the draw invalid and its result is undefined on every card. Show with a fixture whether the runtime makes such a draw, and if it does, make the pipeline again for the formats bound at the draw, kept per format set.
- **What is known, and how:** suspected, from reading another public fork that remakes its pipelines this way; not verified here. By reading the code: the host pipeline takes its formats from the title's description (`RTVFormats` and `DSVFormat` in `crates/victory-gpu/src/pipeline/host.rs`), and the draw path does not compare them with the targets bound.
- **Where the code is:** `crates/victory-gpu/src/pipeline/host.rs` (where the description is made); `crates/victory-gpu/src/draw.rs` (`bind` and the draw path); `crates/victory-gpu/src/pipeline/library.rs` (a pipeline's name in the library holds its graphics state).
- **Files owned:** `crates/victory-gpu/src/pipeline/host.rs`, `crates/victory-gpu/src/draw.rs`, a test file under `crates/victory-gpu/src/`
- **Ruled out:** —
- **Done when:** a fixture on the software adapter with the debug layer makes a pipeline for one target format and draws while a target of another is bound, and reports whether the layer calls it invalid. If it does: the fix, behind a switch that is off, makes and keeps a pipeline per bound format set; with the switch on the fixture passes, with it off the default path is unchanged byte for byte; the GPU crate's tests pass on the software adapter. Either way, the comment says what the fixture showed.
- **Test without the title:** yes: the fixture needs no copy of the title.
- **Judged by:** the Direct3D 12 debug layer in the fixture; in a run, PIX's debug layer on a captured frame.
- **Out of scope:** counting such draws in a run (a maintainer holds that diagnostic); sample-count mismatches.
- **Size:** M
- **Needs:** none.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
