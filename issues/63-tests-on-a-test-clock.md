# 63 — GPU crate tests that bound real elapsed time

- **Title:** Find the GPU crate's tests that bound real elapsed time, and move them to a clock the test controls
- **Area:** Tools and measurement
- **Cards:** all
- **Goal:** Tests that sleep or compare `Instant::now()` against a bound pass or fail with the machine's load, not with the code. The frame budget's timing tests were moved to a clock the test moves by hand and now pass under load. List the other tests in `victory-gpu` that bound real elapsed time, then move the ones that test arithmetic or ordering onto a test clock, one per commit.
- **What is known, and how:** measured by repeated test runs on a deliberately loaded machine: before their fix the budget's timing tests failed about one run in three; after it, five loaded runs passed. By a search of the code, sleeps and `Instant::now()` appear in at least `host/tests/fetch.rs`, `host/tests/wide.rs` and `texture_pack/tests.rs`; a search is a lower bound, and the list is part of the task.
- **Where the code is:** `crates/victory-gpu/src/budget.rs` and `budget_tests.rs` (the test clock to follow); the test files above and any others the list finds.
- **Files owned:** test code under `crates/victory-gpu/src/` that the list names; production code only where a clock must be passed in, and then unchanged in behaviour
- **Ruled out:** tests whose point is a real timeout (a wait that must end): those keep a generous bound and say why in a comment.
- **Done when:** the task comment lists every test found, each marked "moved", "kept, a real timeout" or "kept, why"; each moved test is shown to fail when the code it guards is broken (say how); the GPU crate's tests pass on the software adapter five times in a row while another heavy build runs.
- **Test without the title:** yes, all of it.
- **Out of scope:** the translator's tests; changing what a test checks.
- **Size:** M
- **Needs:** none.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
