# 52 — Log the ray-tracing structure builder's calls

- **Title:** Log the title's calls into its acceleration-structure library, forwarded unchanged
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** The title builds its ray-tracing structures through a small set of imports from a library it ships. Serve those imports with a shim that logs each call and forwards it unchanged to the title's own library, in diagnostics builds only, so the structure work can start from the call mix, the geometry formats and whether primitive identity arrives.
- **What is known, and how:** by reading the code: the imports are few and are logged dynamically by this task; no run has logged them yet. The runtime does not build ray-tracing structures on any card yet, and serving a title library while forwarding to the title's own copy does not exist in the runtime; this task adds it.
- **Where the code is:** `apps/victory/src/run.rs` (`title_dll`, which decides whether a library the title imports is served by the runtime or loaded from the title's folder); `crates/victory-hle/src/registry.rs` (the export table); `crates/victory-hle/src/modules/mod.rs` (registering a module); a new `crates/victory-hle/src/modules/bvh_builder.rs`.
- **Files owned:** the four files named above
- **Ruled out:** building DXR structures; changing the library's behaviour; logging every call forever.
- **Done when:** with `VICTORY_BVH_BUILD_LOG=1` in a diagnostics build, the first calls print in full and then one summary a second (calls by kind, elements, sizes), addresses as an allocation plus an offset, no vertex or index data; without it the library loads exactly as today. Tests cover a served library whose exports all resolve, one whose export is missing (not served), and the record parser on a synthetic 0x50-byte element; `cargo xtask policy` passes. With your own copy, the log shows the build and rehydrate calls of a streaming minute and the title runs as it did without the shim.
- **Test without the title:** the served and unserved fixtures and the parser test use made-up data.
- **Out of scope:** the rehydrate note (issue 54); DXR; the builder's on/off switch.
- **Size:** L
- **Needs:** your own copy of the title and a run that streams the world; the unit tests need none.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
