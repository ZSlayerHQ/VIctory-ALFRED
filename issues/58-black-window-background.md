# 58 — Paint the runtime's window black from the start

- **Title:** Give the runtime's window a black background, so launch shows no white flash
- **Area:** CPU side: speed and memory
- **Cards:** all
- **Goal:** Between the window opening and the first frame presented, the runtime's window shows white. Give its window class a black background so the screen stays dark until the title draws.
- **What is known, and how:** by reading the code: `make_window` registers the class with no background brush (`hbrBackground` is left empty) and `window_proc` hands every message to `DefWindowProcW`.
- **Where the code is:** `crates/victory-gpu/src/host.rs`: `make_window` and `window_proc`.
- **Files owned:** `crates/victory-gpu/src/host.rs` (those two functions only); a test beside them
- **Ruled out:** the swap chain and the title's own window calls; the white is before either draws.
- **Done when:** a test creates the window and reads its class background (`GetClassLongPtrW` with `GCLP_HBRBACKGROUND`) as the stock black brush, and fails without the change; nothing else about the window changes. With your own copy, a recording of a launch shows no white frame before the first picture.
- **Test without the title:** yes: creating a window needs no card and no copy of the title.
- **Judged by:** a recording of the launch, with ffmpeg's `blackframe` filter behind any figure you give.
- **Out of scope:** window size, title text and the caption (issue 19).
- **Size:** S
- **Needs:** none.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
