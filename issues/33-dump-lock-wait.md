# 33 — Release the submission lock before waiting for a target dump

- **Title:** Release the submission lock before waiting for a target dump
- **Area:** Tools and measurement
- **Cards:** all
- **Files owned:** crates/victory-gpu/src/host.rs
- **Symptom:** The lock spans list allocation, recording and the queue fence for every target, so another frame's submission or the retirement helper can wait through older GPU work ahead of the dump.
- **What is known, and how:** by reading the code: a review found that the lock spans each target's allocation, recording and queue fence, so the wait keeps it held.
- **Ruled out:** —
- **Done when:** State read, barrier recording and enqueue stay serialized, and the wait happens outside the lock on the dump's own fence.
- **Test without the title:** the test in "Done when" runs on the software adapter (`VICTORY_WARP=1 VICTORY_D3D12_WARP=1 cargo test -p victory-gpu --features diagnostics --locked`); no copy of the title is needed.
- **Size:** M
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
