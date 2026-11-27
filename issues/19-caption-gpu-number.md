# 19 — Make the caption's GPU figure agree with PresentMon

- **Title:** Make the caption's GPU figure a load that agrees with PresentMon, or label it as a wait
- **Area:** Tools and measurement
- **Cards:** all
- **Goal:** The window caption shows a GPU busy figure. Make it the card's busy share as PresentMon measures it (`MsGPUBusy` over the frame time), or label it plainly as the time the runtime waited for the card.
- **What is known, and how:** by reading the code and comparing: the figure was built from the runtime's own waits for the card, and once disagreed by a wide margin with an independent GPU clock in the same frame. The runtime's own frame timings and GPU timestamps were since found to overlap and not sum, so they are not a judge.
- **Where the code is:** `crates/victory-gpu/src/host.rs`: `caption`, `caption_on`, `caption_due`, and where the figures passed to `caption` are made; `crates/victory-gpu/src/stamps.rs` for the card's timestamps.
- **Files owned:** `crates/victory-gpu/src/host.rs` (the caption only), `crates/victory-gpu/src/stamps.rs`
- **Done when:** a unit test feeds made-up figures and asserts the caption's text; with your own copy, a run with PresentMon shows the caption's figure within 5 points of PresentMon's GPU busy share over the same seconds, or the caption says "wait" and what it divides.
- **Test without the title:** the caption text test runs on the software adapter; the agreement check needs your own copy.
- **Judged by:** PresentMon (`MsGPUBusy`, or `GPUBusy` in newer versions).
- **Out of scope:** the live stats file's fields; the frame budget lines.
- **Size:** S
- **Needs:** your own copy for the agreement check; PresentMon.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
