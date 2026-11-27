# 30 — The target-dump refusal can stall recording

- **Title:** The target-dump refusal can stall recording
- **Area:** Tools and measurement
- **Cards:** all
- **Files owned:** crates/victory-gpu/src/capture.rs
- **Symptom:** The target-dump refusal writes its warning to stderr inside OnceLock initialization. A render-target bind can enter it while holding list state, and other gate callers wait for initialization, so a blocked stderr sink can stall the title on its first refused request.
- **What is known, and how:** by reading the code: a review found the stall on the render-target bind's call path.
- **Ruled out:** —
- **Done when:** The gate is evaluated and the refusal logged during diagnostics startup, before command recording and presentation, with a test for a blocked sink.
- **Test without the title:** the test in "Done when" runs on the software adapter (`VICTORY_WARP=1 VICTORY_D3D12_WARP=1 cargo test -p victory-gpu --features diagnostics --locked`); no copy of the title is needed.
- **Size:** S
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
