# 25 — An interior blows out to white, with hard-edged light slabs

- **Title:** In an interior, characters blow out to white and hard-edged slabs of light lie across surfaces
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** In some interiors, characters' clothing and skin blow out to white (in close-ups nearly solid white with glowing outlines), and hard-edged slabs of coloured light lie across the floor and counters. Name the pass and the value where the tone departs, then fix it.
- **What is known, and how:** seen on recordings of runs. The tiled-lighting and clustering programs translate and run. Other interiors show weaker patches of the same kind. Nothing is known about the cause.
- **Where the code is:** to be confirmed by a maintainer once the pass is named.
- **Files owned:** none until the pass is named.
- **Done when:** a capture names the pass and the target where the white or the slab first appears, and what it reads (formats, sizes and pass names in the comment; no pictures); then a fix with a fixture, and a recording of such an interior at normal tone.
- **Test without the title:** the first step needs your own copy and a save in an affected interior; the fix's fixture does not.
- **Judged by:** PIX or RenderDoc to name the pass; ffmpeg's `ssim` on a recording for "fixed".
- **Out of scope:** the washed-out start (issue 14).
- **Size:** M
- **Needs:** your own copy of the title, with a save in an affected interior; a capture tool.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
