# 38 — Serialize present selection and the texture-state round trip for target dumps

- **Title:** Serialize present selection and the texture-state round trip for target dumps
- **Area:** Tools and measurement
- **Cards:** all
- **Files owned:** crates/victory-gpu/src/capture.rs, crates/victory-gpu/src/queue.rs, crates/victory-gpu/src/resource.rs
- **Symptom:** A concurrent present can publish the copy-source state before its GPU transition is queued, so a target dump can skip the required source barrier; a later present reaching frame selection first can also discard the earlier frame's targets.
- **What is known, and how:** by reading the code: a review found that present selection can publish the copy-source state before its transition is queued, and that frame selection can discard the earlier frame's targets.
- **Ruled out:** —
- **Done when:** Frame selection is serialized with present numbering, and the present's state round trip is recorded and submitted under the submission lock; failing that, target dumps stay disabled for landing.
- **Test without the title:** the test in "Done when" runs on the software adapter (`VICTORY_WARP=1 VICTORY_D3D12_WARP=1 cargo test -p victory-gpu --features diagnostics --locked`); no copy of the title is needed.
- **Size:** M
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
