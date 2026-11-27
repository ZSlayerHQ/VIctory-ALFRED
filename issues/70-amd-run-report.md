# 70 — The default build on an AMD card

- **Title:** Run the default build on an AMD card and report what differs from the known faults
- **Area:** Tools and measurement
- **Cards:** amd
- **Goal:** The runtime takes a different path on a 64-lane device with a large view heap (AMD) than on a 32-lane device with the smaller heap (NVIDIA): console waves at their own width, and view heaps of 2^21 views. Run the default build on an AMD card at the larger console's profile and report how it fares, so the other vendor's path has evidence of its own.
- **What is known, and how:** by reading the code (CONTRIBUTING.md, "Two vendors, one codebase"). The project's recent measurements are from NVIDIA cards; sibling forks report runs on AMD cards with their own builds.
- **Where the code is:** no change; the run's own log names the path taken: the `host adapter:` line and the line "host view heaps hold <n> views".
- **Files owned:** none; a run report.
- **Done when:** a run report (the run-report template) with the machine line, the `host adapter:` and view heap lines from the log, the warm-up's refused count and their reasons (compared with the groups KNOWN-FAULTS.md lists), frame times from PresentMon over two minutes of the settled scene, and, for each fault in KNOWN-FAULTS.md you passed, "seen", "not seen" or "not reached". A device loss or hang is reported with the crash breadcrumbs' summary.
- **Test without the title:** no: this is a run with your own copy.
- **Judged by:** PresentMon for frame times; the runtime's warm-up list for refusals.
- **Out of scope:** fixing what the run shows: open a fault for each new fault.
- **Size:** S
- **Needs:** your own copy of the title; an AMD RDNA2 or later card; PresentMon.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
