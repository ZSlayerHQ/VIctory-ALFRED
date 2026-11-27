# 41 — Lower the primitive-ordered preamble

- **Title:** Lower the primitive-ordered preamble
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** A group of pixel programs uses the console's primitive-ordered pixel shading, an ordering of overlapping pixels the translator has no lowering for; each opens with a preamble that needs a hardware input the translator lacks. Recognise and drop that preamble and declare the views it writes rasterizer-ordered, behind a check of the device's support, so the group translates and draws.
- **What is known, and how:** the warm-up's refusal list (the runtime's own count): a group of pixel programs is refused with "a hardware SGPR input is not supported"; the refusal names the missing input. The preamble cannot simply be dropped: it orders overlapping pixels and the images they write.
- **Where the code is:** `crates/victory-shader/src/emit/lower.rs` `Lower::inputs` (the `Val::Missing("a hardware SGPR input")` arm); `crates/victory-shader/src/io.rs` (`Input` and the input raise); `crates/victory-shader/src/emit/shader.rs` (the root UAV metadata's rasterizer-ordered field); `crates/victory-gpu/src/host.rs` (the host device's feature queries); `crates/victory-gpu/src/device.rs` (`OPTIONS` and `check_feature_support`, which is only what the console is told).
- **Files owned:** the five files named above.
- **Ruled out:** a missing instruction lowering; a missing sampler; the view bindings.
- **Done when:** a synthetic fixture with the preamble fails before with "a hardware SGPR input" and translates after on a host that reports rasterizer-ordered views, and refuses with one clear reason on a host that does not; `VICTORY_WARP=1 cargo test -p victory-shader --locked`, the GPU crate's tests on the software adapter and both clippy runs pass; with your own copy, the warm-up refuses fewer programs than your own count before the change, by the size of that group, and a run draws what those programs draw.
- **Test:** a fixture for the preamble and one for a host without support; the first must fail before the change.
- **Test without the title:** yes, both fixtures; the warm-up and the run need your own copy.
- **Out of scope:** any other hardware-input program that differs; the sampler programs (issue 40); the wide-wave group (issue 42).
- **Size:** L
- **Needs:** your own copy of the title, and a card that reports rasterizer-ordered views, to confirm.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
