# 57 — Rotate and compress old run logs

- **Title:** Rotate and compress old run logs
- **Area:** Tools and measurement
- **Cards:** all
- **Goal:** The logs folder grows without limit (past a gigabyte on the maintainers' machine, with single logs over 100 MB). Add a tool that compresses old run logs and their summaries, keeps the newest and any log still cited, verifies each archive before removing the original, and never touches a run in progress.
- **Where the code is:** a new `harness/rotate-logs.mjs` and `harness/rotate-logs.test.mjs`; `harness/disk-guard.mjs` if a scheduled check belongs there; `harness/run-world.sh` names the folder as `$TOOLS/logs`.
- **Files owned:** the new script and test; `harness/disk-guard.mjs` only if it calls the tool
- **Symptom:** No rotation. Old logs stay uncompressed; the disk guard sees the total only when it is too late.
- **What is known, and how:** by reading the harness: nothing rotates or compresses the logs folder; a logging audit asked to compress logs older than a day and keep the ones a run report cites.
- **Ruled out:** Deleting logs; uploading them; changing the log format.
- **Done when:** `node --test harness/rotate-logs.test.mjs` passes; `node harness/rotate-logs.mjs --dry-run <dir>` lists what it would compress and why; on a temporary folder with old and new logs, the old ones become `.gz` (a round-trip read matches) and the newest and the keep-listed stay; a folder with `run.lock` is refused. `node --test "harness/**/*.test.mjs"` passes.
- **Test:** The new test: temporary folders, fixed modification times, a keep list, a lock file, and a gzip round trip.
- **Test without the title:** the new test needs no card and no copy of the title.
- **Out of scope:** The disk guard's thresholds; the summary format; deleting anything.
- **Size:** S
- **Needs:** none.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
