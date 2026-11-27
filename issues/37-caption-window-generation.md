# 37 — Bind caption updates to a window generation

- **Title:** Bind caption updates to a window generation
- **Area:** Tools and measurement
- **Cards:** all
- **Files owned:** crates/victory-gpu/src/host.rs
- **Symptom:** The caption's owner checks compare thread and window ids. They can accept a handle recycled to another window on the same thread, and the handle can change between the check and the write; the presenter also refreshes the owner from its cached handle.
- **What is known, and how:** by reading the code: a review found that the owner checks can accept a recycled window handle.
- **Ruled out:** —
- **Done when:** The caption is bound to a window generation when the presenter is created and checks it on the owner thread at the write; a test covers a recycled handle.
- **Test without the title:** the test in "Done when" runs on the software adapter (`VICTORY_WARP=1 VICTORY_D3D12_WARP=1 cargo test -p victory-gpu --features diagnostics --locked`); no copy of the title is needed.
- **Size:** S
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
