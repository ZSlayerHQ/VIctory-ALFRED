# 69 — Shadowed walls darken after about 35 seconds at rest

- **Title:** Find why walls in shadow darken after the camera has been still for about 35 seconds
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** With the camera still, walls in shadow darken after about 35 seconds. Find what accumulates or changes over that time (a buffer filled over frames, a texture refreshed late, a value that drifts), then fix it.
- **What is known, and how:** seen on recordings of runs with the camera at rest. Nothing is known about the cause.
- **Where the code is:** to be confirmed by a maintainer once the pass is named.
- **Files owned:** none until the pass is named.
- **Done when:** captures at about 5 seconds and about 40 seconds of rest name the pass whose output darkens and the input that changed between them (formats and first differing values; no pictures); then a fix with a fixture, and a recording that holds its tone for two minutes at rest.
- **Test without the title:** finding it needs your own copy; the fixture does not.
- **Judged by:** PIX or RenderDoc for the two captures; ffmpeg's `ssim` between the first and the last minute of the recording for "fixed".
- **Out of scope:** the washed-out start (issue 14).
- **Size:** M
- **Needs:** your own copy of the title; a capture tool.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
