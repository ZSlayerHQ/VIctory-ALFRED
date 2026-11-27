# 54 — Note what the rehydrate call receives

- **Title:** Note what the rehydrate call receives
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** The rehydrate call is where the static world's geometry would be handed to a DXR build. With the logging shim of issue 52 on, record what it actually receives in a run that streams the world, and write the note the first DXR task will start from.
- **What is known, and how:** by reading the library's records: the shape of the call is known; no run has shown what the title passes (batch sizes, source structures, geometry kinds and formats, whether primitive identity is present). Two ideas for the static world wait on this answer: build the card's structure at the rehydrate call, and keep built structures keyed by their packed block's contents.
- **Where the code is:** `crates/victory-hle/src/modules/bvh_builder.rs` (issue 52: the rehydrate wrapper and its record parser); a new `victory/docs/design/rehydrate-inputs.md`.
- **Files owned:** the shim's rehydrate parser and the new note; no behaviour change to the title
- **Ruled out:** the build calls (issue 52); dynamic objects; the trace programs.
- **Done when:** the note has a table of what one streaming run received: calls, elements per call, source and destination sizes, stream records, geometry kinds and flags, index and vertex formats and counts, and whether primitive identity is present (counts and formats, no geometry). A unit test parses a synthetic rehydrate batch into the same table's fields; `cargo test -p victory-hle --locked bvh_builder` passes.
- **Test without the title:** the parser test, on made-up values; the run is the evidence.
- **Out of scope:** building DXR structures; changing the rehydrate; deciding the static-world route.
- **Size:** S
- **Needs:** issue 52 merged; your own copy of the title and a run that streams the world.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
