# 65 — A PIX capture recipe for the harness

- **Title:** Add a PIX recipe to the harness beside the RenderDoc and Nsight ones
- **Area:** Tools and measurement
- **Cards:** all
- **Goal:** The harness can start a run under RenderDoc (`--renderdoc --capture-after-ms`) or Nsight's GPU Trace (`--gpu-trace --trace-after-ms`), but has no recipe for PIX, Microsoft's Direct3D 12 tool and the one that works on every vendor's card. Add `--pix` with a delay, built on `pixtool`, that takes a GPU capture (and, if `pixtool` allows it, a timing capture) of the settled scene.
- **What is known, and how:** by reading the harness: `harness/renderdoc.mjs` and `harness/gpu-trace.mjs` build their commands as pure functions with tests, and `harness/run-world.sh` wires `--renderdoc` and `--gpu-trace`; `harness/tools-check.mjs` already finds `pixtool.exe`. Neither existing recipe has made a real capture yet. The runtime names its own GPU work with debug events (`crates/victory-gpu/src/events.rs`), which PIX shows.
- **Where the code is:** `harness/renderdoc.mjs` and its test (the pattern to follow); `harness/run-world.sh` (the flags and their checks); `harness/tools-check.mjs`.
- **Files owned:** a new `harness/pix.mjs` and `harness/pix.test.mjs`; `harness/run-world.sh` (the new flag only); `docs/DIAGNOSTICS.md` (the PIX section)
- **Ruled out:** opening captures automatically; any change to the other two recipes.
- **Done when:** `node --test "harness/**/*.test.mjs"` and `bash -n harness/run-world.sh` pass, with tests for the command built, a missing `pixtool`, a missing delay, and `--pix` refused beside `--renderdoc` or `--gpu-trace`; with your own copy, one real capture is taken after the delay and the comment says what PIX showed (pass names and counts, no pictures), including whether the runtime's own events appear.
- **Test without the title:** the command-building tests; the real capture needs your own copy.
- **Out of scope:** RenderDoc's and Nsight's first real captures (a maintainer holds them); forwarding the title's own markers.
- **Size:** M
- **Needs:** PIX on Windows (docs/DIAGNOSTICS.md); your own copy for the real capture.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
