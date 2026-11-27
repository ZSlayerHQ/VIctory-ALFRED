# 45 — Keep each DirectStorage read's name and tag

- **Title:** Keep each DirectStorage read's name and tag
- **Area:** CPU side: speed and memory
- **Cards:** all
- **Goal:** A DirectStorage read carries a tag (for cancellation) and a name (the title's own label). The failure line prints both, but a successful read loses them: the walls trace's line has the file, offset and sizes only. Copy the name when the request is enqueued and keep the tag with the read, so every trace line and every summary can name the request.
- **Where the code is:** `crates/victory-hle/src/modules/dstorage.rs`: `Request` (`tag`, `name`), `Read::new`, `WallsLine`, `walls_request`, `walls_start`, and the failure line in `execute_now`.
- **Files owned:** `crates/victory-hle/src/modules/dstorage.rs`
- **Symptom:** The title may free the name's buffer after it enqueues the request; a trace line built later reads no name. The tag is in the request but is not in `WallsLine`.
- **What is known, and how:** by reading the code: `Request` carries a `tag` and a `name`; `execute_now` prints both only on failure.
- **Ruled out:** Cancellation by tag, which the queue does not implement; the file path, which the trace already has.
- **Done when:** A test enqueues a request with a tag and a name, runs it, and finds both on the trace line (and on the failure line); a null name prints an empty name, not a crash; `cargo test -p victory-hle dstorage` and clippy pass.
- **Test:** `dstorage.rs` tests: the existing request builder with `tag` and `name` set; read the line from the test sink.
- **Test without the title:** `cargo test -p victory-hle --locked dstorage` needs no card and no copy of the title.
- **Out of scope:** Implementing cancellation; changing the request layout; the walls trace's file name.
- **Size:** S
- **Needs:** none.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
