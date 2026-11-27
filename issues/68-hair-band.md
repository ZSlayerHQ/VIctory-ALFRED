# 68 — A dark band across hair

- **Title:** Name the program that draws a dark band across characters' hair
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all (say which card you saw it on; a 32-lane and a 64-lane device may differ)
- **Goal:** A dark band runs across characters' hair. Name the pass and program that make it, then fix the translation or binding at fault.
- **What is known, and how:** seen on recordings of runs on an NVIDIA card. Suspected, not shown: a 64-lane program behind the hair's mask, which on a 32-lane device runs in one of the 32-lane forms (CONTRIBUTING.md, "Two vendors, one codebase"). If the band is absent on a 64-lane device, that points the same way.
- **Where the code is:** to be confirmed by a maintainer once the program is named; the wave forms are in `crates/victory-shader/src/emit/paired.rs` and `emit/mod.rs`.
- **Files owned:** none until the program is named.
- **Done when:** the comment names the program (16-hex id), the form it was translated in, and the value that differs from a correct pixel; a fixture reproduces it on the software adapter, then the fix, and a recording shows the hair without the band on the card you saw it on.
- **Test without the title:** finding it needs your own copy; the fixture does not.
- **Judged by:** PIX or RenderDoc to name the program; a recording for "fixed".
- **Out of scope:** the wide-wave refusals of issue 42.
- **Size:** M
- **Needs:** your own copy of the title; a capture tool; a second vendor's card helps.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
