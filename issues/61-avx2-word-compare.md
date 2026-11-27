# 61 — A wide word compare equal to the scalar one

- **Title:** Compare title memory against what the card wrote with AVX2, giving exactly the scalar result
- **Area:** CPU side: speed and memory
- **Cards:** all
- **Goal:** After a submission, the runtime compares title memory as it read it with the bytes merged from the card's writes, and sends back the 4-byte words that differ (`word_runs`). It walks the bytes on the submitting thread. Write an AVX2 version that returns exactly the same runs, as a pure function behind a switch that is off (`VICTORY_UNDER_WIDE=1`), so the compare can later move off the submitting thread.
- **What is known, and how:** by reading the code: `word_runs` passes equal bytes 32 at a time (`first_difference`), then grows each run word by word with a slice compare. Another public fork does this compare with AVX2 words and reports a saving on its own scenes; nothing is measured here.
- **Where the code is:** `crates/victory-gpu/src/shadow.rs`: `word_runs`, `first_difference`, `changed_runs`, and the tests in `mod difference_tests`.
- **Files owned:** `crates/victory-gpu/src/shadow.rs` (the new function, its switch and its tests only)
- **Ruled out:** threads for the compare, and sending pairs of words: later steps that a maintainer holds.
- **Done when:** a test checks the AVX2 function against `word_runs` on random buffers, on runs at every alignment of the address modulo 4 and 32, on empty and one-byte slices and on slices of unequal length, and they agree on every case; with the switch off nothing calls the new function and the default path is unchanged byte for byte; the GPU crate's tests pass on the software adapter.
- **Test without the title:** yes: the function is pure.
- **Judged by:** the equality test for correctness. A speed claim gives the submitting thread's time before and after from a system trace (WPA) or Tracy, on the same scene.
- **Out of scope:** moving the compare to other threads; the 4,096-send limit; the merge itself.
- **Size:** S
- **Needs:** none; a processor with AVX2 (any that runs the title).
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
