# 53 — DXR tier probe fixture

- **Title:** DXR tier probe fixture
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** Add a fixture that queries the host device for DirectX Raytracing tier 1.1 and runs a trivial dispatch through the runtime's own list and queue path, so the first ray-tracing task starts from a known-capable machine. Skip on a card below tier 1.1.
- **Where the code is:** `crates/victory-gpu/src/host.rs` (`create`, the `CheckFeatureSupport` calls, `test_host`), with a new test beside the host tests; `crates/victory-gpu/src/list.rs` test helpers for a dispatch.
- **Files owned:** `crates/victory-gpu/src/host.rs` (the query and the test), `crates/victory-gpu/src/list.rs` (test helper only)
- **Symptom:** No line in the repository says which DXR tier the host device reports; a failure in a later trace task cannot be told from a machine that cannot run it.
- **What is known, and how:** by reading the code: The host queries options 1, 2, 7 and 12 but not `D3D12_FEATURE_D3D12_OPTIONS5`; the scope notes call a tier 1.1 probe cheap and unrun.
- **Ruled out:** Building acceleration structures; translating trace programs; DLSS.
- **Done when:** `cargo test -p victory-gpu dxr_tier -- --nocapture` prints the adapter, the tier and one line from a dispatch through the queue, and passes by skipping below tier 1.1; on an RTX 5090 it reports tier 1.1 and the dispatch. `cargo clippy` is clean.
- **Test:** The fixture itself; a dispatch is the exercise of the binding path.
- **Test without the title:** the fixture runs on any adapter and skips below tier 1.1; a card that reports tier 1.1 shows the positive case.
- **Out of scope:** Ray queries in shaders; structure builds; performance.
- **Size:** S
- **Needs:** none; a DXR-capable GPU to see the positive case.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
