# 36 — Count a direct target dump as card work

- **Title:** Count a direct target dump as card work
- **Area:** Tools and measurement
- **Cards:** all
- **Files owned:** crates/victory-gpu/src/host.rs
- **Symptom:** The direct dump submission bypasses the card-sent accounting, so the GPU clock can count the dump as idle.
- **What is known, and how:** by reading the code: a review found that the direct dump bypasses the card-sent accounting.
- **Ruled out:** —
- **Done when:** The card-sent count is updated after enqueueing the dump, as a normal submission does; a test covers the clock.
- **Test without the title:** the test in "Done when" runs on the software adapter (`VICTORY_WARP=1 VICTORY_D3D12_WARP=1 cargo test -p victory-gpu --features diagnostics --locked`); no copy of the title is needed.
- **Size:** S
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
