# 39 — The dump switch's row must state the unsafe requirement

- **Title:** The dump switch's row must state the unsafe requirement
- **Area:** Tools and measurement
- **Cards:** all
- **Files owned:** `victory/README.md` (the row for `VICTORY_DUMP_TARGETS` in the table of switches)
- **Symptom:** The row for saving render targets omits that it needs `VICTORY_DUMP_UNSAFE=1` and why: the dump can race a present's change of a texture's state and record an invalid barrier.
- **What is known, and how:** by reading the code: a review found that the README row omits the unsafe switch and the barrier risk.
- **Ruled out:** —
- **Done when:** The row states the required switch and the barrier risk in one sentence.
- **Test without the title:** none needed: a documentation change; check the sentence against `dump_on` and `dump_allowed` in `crates/victory-gpu/src/capture.rs`.
- **Size:** S
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
