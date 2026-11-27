# 34 — Reporter startup and rejected-count races

- **Title:** Reporter startup and rejected-count races
- **Area:** Tools and measurement
- **Cards:** all
- **Files owned:** crates/victory-gpu/src/host.rs
- **Symptom:** The receiver can finish its last job before a producer increments a rejected count, then block in receive; a worker spawn failure also leaves the one-time initialization set to none for good.
- **What is known, and how:** by reading the code: a review found the last-job increment race and the one-time initialization left set to none.
- **Ruled out:** —
- **Done when:** A timed receive checks the counter, and worker startup is retried off the completion path; tests cover both.
- **Test without the title:** the test in "Done when" runs on the software adapter (`VICTORY_WARP=1 VICTORY_D3D12_WARP=1 cargo test -p victory-gpu --features diagnostics --locked`); no copy of the title is needed.
- **Size:** S
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
