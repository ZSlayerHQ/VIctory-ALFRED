# 67 — Characters lit wrongly for single frames

- **Title:** Find the buffer behind characters' lighting that changes for single frames
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** Characters are lit very differently for a single frame now and then, then return to normal. Find the input of their lighting that holds other values in those frames (a buffer present or current only some frames), then fix what makes it so.
- **What is known, and how:** seen on recordings of runs. Not yet known: which pass or buffer; nothing has been ruled out.
- **Where the code is:** to be confirmed by a maintainer once the buffer is named. Candidates to read first: how a buffer the card writes is brought back for the next frame (`crates/victory-gpu/src/shadow.rs`, `host/relay.rs`) and how a texture's host copy is refreshed (`crates/victory-gpu/src/texture.rs`).
- **Files owned:** none until the buffer is named.
- **Done when:** captures of a flipped frame and of a normal frame next to it name the lighting pass and the input that differs between them (formats, sizes, the first differing values; no pictures); then a fix with a fixture, and a recording with no flip over the same stretch.
- **Test without the title:** no: finding it needs your own copy; the fix's fixture would not.
- **Judged by:** PIX or RenderDoc for the two frames; ffmpeg's `freezedetect` or `ssim` over a recording for "no flip".
- **Out of scope:** the steady character lighting faults of issues 23 and 25.
- **Size:** M
- **Needs:** your own copy of the title; PIX or RenderDoc.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
