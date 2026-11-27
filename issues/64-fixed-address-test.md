# 64 — A watch test that loses its address to another test

- **Title:** Make the write-watch test that reuses a freed address independent of other tests
- **Area:** CPU side: speed and memory
- **Cards:** all
- **Goal:** `write_watched_memory_reused_at_a_guarded_view_is_asked_about` unmaps a view and then asks for new memory at the same address. Tests run in parallel, so another test can take that address first and the test fails now and then. Make it hold its own range, or make the tests that name an address share one lock, so it cannot lose the race.
- **What is known, and how:** seen once in a plain-build test run and not in 14 runs since; by reading the test, the address is free between `UnmapViewOfFile` and `VirtualAlloc` at that address. A race in the test, not in the code it checks; the public CI will meet it.
- **Where the code is:** `crates/victory-gpu/src/watch.rs`, the test `write_watched_memory_reused_at_a_guarded_view_is_asked_about` and any other test in that file that allocates at a chosen address.
- **Files owned:** the tests in `crates/victory-gpu/src/watch.rs`
- **Ruled out:** a fault in the write watch itself.
- **Done when:** the test still checks what it checks (write-watched memory at a once-guarded address is asked about, not answered clean); a stress run of the crate's tests (`--test-threads` at the machine's core count, 50 times) shows no failure; the other tests that choose an address are listed in the comment with what was done for each.
- **Test without the title:** yes.
- **Out of scope:** the guard and write-watch code.
- **Size:** S
- **Needs:** none.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
