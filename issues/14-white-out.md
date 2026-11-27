# 14 — Find the white-outs and the washed-out start

- **Title:** Find what washes out the world's first 30 seconds and what whites out the screen
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** The first 30 seconds or so of the world look washed out, and at times the whole screen goes white. Name the pass and the value that produce them, then fix it.
- **What is known, and how:** seen on recordings of runs at the larger console's profile on an NVIDIA card. Nothing is known about the cause; no pass has been named.
- **Where the code is:** to be confirmed by a maintainer once the pass is named; the translator (`crates/victory-shader`) or the textures and descriptors (`crates/victory-gpu/src/texture.rs`, `descriptor/`).
- **Files owned:** none until the pass is named; then the files the fix needs, agreed on the task.
- **Done when:** a capture names the pass and the target where the white first appears, and the value it reads (counts, formats and pass names in the comment; no pictures); then a fix, with a fixture for the translation or texture fault it turns out to be, and a recording shows the start and the scene at their normal tone.
- **Test without the title:** the first step needs your own copy; the fix's fixture does not.
- **Judged by:** PIX or RenderDoc for naming the pass; a recording for "fixed", with ffmpeg's filters behind any figure you give (docs/DIAGNOSTICS.md).
- **Out of scope:** the interior fault of issue 25; the title's exposure options.
- **Size:** M
- **Needs:** your own copy of the title; a capture tool.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
