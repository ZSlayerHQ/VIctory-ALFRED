# 35 — Keep the recorder's refusal logs off the list-state lock

- **Title:** Keep the recorder's refusal logs off the list-state lock
- **Area:** Tools and measurement
- **Cards:** all
- **Files owned:** crates/victory-gpu/src/capture.rs, crates/victory-core/src/logging.rs, crates/victory-gpu/src/list.rs
- **Symptom:** Oversized or over-budget captures log synchronously under the recorder's list-state lock. With the trace target enabled, a blocked stderr sink can stop recording on those paths.
- **What is known, and how:** by reading the code: a review found that the refusal logs are written while the recorder's list-state lock is held.
- **Ruled out:** —
- **Done when:** The small refusal line goes through the same bounded reporter, and rejections are counted there; a test covers a blocked sink.
- **Test without the title:** the test in "Done when" runs on the software adapter (`VICTORY_WARP=1 VICTORY_D3D12_WARP=1 cargo test -p victory-gpu --features diagnostics --locked`); no copy of the title is needed.
- **Size:** S
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
