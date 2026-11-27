# 71 — Measure the off-by-default speed switches on your machine

- **Title:** A/B each off-by-default speed switch with PresentMon on your own machine
- **Area:** CPU side: speed and memory
- **Cards:** all
- **Goal:** Several speed changes are merged switched off, and none has been measured on more than one machine. Measure each against the same build with the switch off, on your own machine, so a decision to switch one on rests on more than one processor and card.
- **What is known, and how:** by reading the code: the switches below are off unless set. The frame is limited by one submitting thread (KNOWN-FAULTS.md, "The frame is limited by one thread"), so a switch's effect may differ by processor. The maintainers' own measurement series is under way; yours adds other hardware.
  - Read once at start, so one run each way: `VICTORY_WIDE_REFRESH=1` (refreshes of host copies on threads), `VICTORY_HASH_WIDE=1` (the card's hash, wide form), `VICTORY_AVOID_CPUS=<processors>` (keeps title threads off the processor that takes the card's interrupts; the trace names it), `VICTORY_D3D12_BACKGROUND=<mode>` (the driver's background shader work).
  - Live, through the control file `VICTORY_LIVE=<file>`, so one run with segments: `VICTORY_FETCH_QUEUE`, `VICTORY_LAND_SCAN`, `VICTORY_UPLOAD_POOL`.
- **Where the code is:** no change. The switches' rows are in the runtime's README; where one has none, its comment is in `crates/victory-gpu/src/host/refresh.rs`, `cpu.rs` or `host/background.rs`.
- **Files owned:** none; a run report per switch, or one report with a table.
- **Done when:** for each switch measured: two runs (or two live segments) on the same build, scene and switches, one with the switch and one without, each with at least two minutes of the settled scene; PresentMon's median and 95th-percentile frame time and GPU busy share for both, the machine line, and anything the picture did differently. Say how many pairs you ran; one pair proves little.
- **Test without the title:** no: these are runs with your own copy.
- **Judged by:** PresentMon (docs/DIAGNOSTICS.md); the runtime's live segment lines may explain a figure, not stand in for it.
- **Out of scope:** changing a default; switches not listed here.
- **Size:** M
- **Needs:** your own copy of the title; PresentMon.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
