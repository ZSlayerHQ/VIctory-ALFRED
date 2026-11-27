# 11 — Bound LDS accesses by the allocation

- **Title:** Bound LDS accesses by the title's allocation
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** The translator takes every LDS (group-shared memory) address as dword-aligned and inside the program's allocation. Give an access outside the allocation the console's result, so a program that strays reads and writes what the console's would, on every card.
- **What is known, and how:** by reading the code: `crates/victory-shader/src/emit/lower.rs` says so in a `shortcut:` comment ("LDS addresses are taken as dword-aligned and in bounds"). A crash report from the driver's shader error reporting (Nsight Aftermath) once placed one compute program's read past its allocation, in no bound resource; the access is fatal only with that reporting on, but out of range in every run. The rule to follow is the RDNA2 ISA guide's for LDS addresses outside the allocation.
- **Where the code is:** `crates/victory-shader/src/emit/lower.rs`: the LDS load, store and atomic lowering beside that comment, and the program's LDS size (`lds_dwords` in the lowering's state).
- **Files owned:** `crates/victory-shader/src/emit/lower.rs`, a new test file beside it
- **Ruled out:** as a cause of a hang (fatal only under error reporting).
- **Done when:** a fixture with an access past the allocation fails before the change and passes after (the read gives the ISA's value, the write leaves LDS unchanged); an in-bounds program translates byte for byte as before; `VICTORY_WARP=1 cargo test -p victory-shader --locked` passes. With your own copy, a run under the driver's shader error reporting shows no LDS fault.
- **Test without the title:** yes: a synthetic GFX10 compute program in the style of `crates/victory-shader/src/emit/wave64_tests.rs`, run on the software adapter.
- **Out of scope:** unaligned LDS addresses; LDS shared across waves without barriers.
- **Size:** M
- **Needs:** none for the fixture; your own copy and an NVIDIA card for the Aftermath run.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
