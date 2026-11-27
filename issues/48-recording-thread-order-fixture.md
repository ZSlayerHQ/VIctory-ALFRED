# 48 — Stress fixture for ordering between recording threads

- **Title:** Stress fixture for ordering between recording threads
- **Area:** CPU side: speed and memory
- **Cards:** all
- **Goal:** Lists are recorded on many threads and run in submission order; at submission, a list's prelude moves each texture it uses from its real state to the one the list starts with, and the real state becomes the one the list ends with. Build a stress fixture that records lists on two or more threads against the same textures and buffer, submits them in both orders, and fails if the preludes or the values read do not follow the submission order.
- **What is known, and how:** by reading the code: the rule is written in `HostList::prelude`'s comment and in the submission path, and is tested one list at a time; nothing stresses two recording threads and a shared resource.
- **Where the code is:** `crates/victory-gpu/src/host.rs` (`HostList::prelude`, the submission path and its tests); `crates/victory-gpu/src/draw.rs` (`replay`); `crates/victory-gpu/src/list.rs` (`detach_parts`); `crates/victory-gpu/src/queue.rs` (`execute` order).
- **Files owned:** test code only, in a new file under `crates/victory-gpu/src/host/tests/`
- **Ruled out:** the title's own thread count; the fixture uses its own threads.
- **Done when:** the fixture runs 50 iterations with two threads and both submission orders, asserts every read sees the value and state its order requires, and passes; an injected change that moves the prelude after its list makes it fail (say how you checked). The GPU crate's tests pass on the software adapter.
- **Test:** the fixture itself; keep it deterministic with barriers, not sleeps.
- **Test without the title:** yes, on the software adapter.
- **Out of scope:** the title's recording; changing the submission lock.
- **Size:** M
- **Needs:** none.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
