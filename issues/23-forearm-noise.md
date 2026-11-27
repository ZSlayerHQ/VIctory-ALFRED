# 23 — A character's forearms render purple or as coloured noise

- **Title:** A character's forearms render purple or as coloured noise
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** One character's forearms render purple, and on others arms show coloured noise where tattoo or decal layers sit over skin, while their faces are right. Find the draw and the binding that differ, then fix it.
- **What is known, and how:** seen on recordings of runs. The skin, decal and tattoo programs translate and run (the warm-up refuses none of them). Not yet known: which draw. One candidate, suspected and not tied to this: the translator reads 16-bit integer values through integer views as zero (KNOWN-FAULTS.md, "Faces: 16-bit integer values read as zero").
- **Where the code is:** the binding trace, `VICTORY_BIND_TRACE=<pixel shader ids>` (`crates/victory-gpu/src/bind_trace.rs`), logs the bindings of the first draws of those pixel programs; the fix's files are to be confirmed by a maintainer.
- **Files owned:** none until the draw is named; `crates/victory-gpu/src/bind_trace.rs` if the trace needs a field.
- **Done when:** the comment names the pixel program (its 16-hex id) and the binding whose texture or view differs from a correct draw of the same program; then a fix with a fixture, and a recording shows the arms' colour matching the face's.
- **Test without the title:** the trace and the capture need your own copy; the fixture for the fault found does not.
- **Judged by:** RenderDoc or PIX for the draw's bindings; a recording for "fixed".
- **Out of scope:** the 16-bit integer view fix itself, which a maintainer holds.
- **Size:** M
- **Needs:** your own copy of the title; RenderDoc or PIX.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
