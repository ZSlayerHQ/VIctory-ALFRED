# 32 — Snapshot a repeated list's records after each execution

- **Title:** Snapshot a repeated list's records after each execution
- **Area:** Tools and measurement
- **Cards:** all
- **Files owned:** crates/victory-gpu/src/host.rs
- **Symptom:** When a list is repeated in one submission and each execution is timed and fenced separately, its buffer is snapped only after all groups, so both timing lines describe the last run.
- **What is known, and how:** by reading the code: a review found that the snapshot is taken only after all groups.
- **Ruled out:** —
- **Done when:** For a repeated list, the snapshot is taken after each execution's fence and before the list is reissued; a test covers two executions with different records.
- **Test without the title:** the test in "Done when" runs on the software adapter (`VICTORY_WARP=1 VICTORY_D3D12_WARP=1 cargo test -p victory-gpu --features diagnostics --locked`); no copy of the title is needed.
- **Size:** S
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
