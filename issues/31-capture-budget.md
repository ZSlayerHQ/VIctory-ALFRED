# 31 — Reserve a shared byte budget before mapping capture readbacks

- **Title:** Reserve a shared byte budget before mapping capture readbacks
- **Area:** Tools and measurement
- **Cards:** all
- **Files owned:** crates/victory-gpu/src/host.rs, crates/victory-gpu/src/capture.rs
- **Symptom:** For listed captures, a slow reporter can retain a very large queued CPU payload (hundreds of slots of multi-megabyte reports, plus one active report), even when the reports are later discarded.
- **What is known, and how:** by reading the code: a review found that the queue slots and per-list limits can retain the payload.
- **Ruled out:** —
- **Done when:** Queue slots and a shared byte budget are reserved before mapping and returned on drop or after reporting; a test covers the budget under a slow reporter.
- **Test without the title:** the test in "Done when" runs on the software adapter (`VICTORY_WARP=1 VICTORY_D3D12_WARP=1 cargo test -p victory-gpu --features diagnostics --locked`); no copy of the title is needed.
- **Size:** M
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
