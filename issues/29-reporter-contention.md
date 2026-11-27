# 29 — A reporter batch can be dropped under sender-mutex contention

- **Title:** A reporter batch can be dropped under sender-mutex contention
- **Area:** Tools and measurement
- **Cards:** all
- **Files owned:** crates/victory-gpu/src/host.rs
- **Symptom:** A producer can drop an entire batch while another producer briefly holds the sender mutex, even with an idle reporter and an empty channel. Recording also takes that mutex for every indirect report or deferred callback, and can wait behind thread creation while holding the list's state.
- **What is known, and how:** by reading the code: a review found that the sender lock and the try-lock paths allow a batch to be dropped.
- **Ruled out:** —
- **Done when:** The started sender is published once and producers read it without locking; startup is retried off the completion path; a test covers the contention.
- **Test without the title:** the test in "Done when" runs on the software adapter (`VICTORY_WARP=1 VICTORY_D3D12_WARP=1 cargo test -p victory-gpu --features diagnostics --locked`); no copy of the title is needed.
- **Size:** M
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
