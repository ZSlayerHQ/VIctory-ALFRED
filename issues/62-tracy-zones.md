# 62 — Tracy zones at the frame budget's phases

- **Title:** Show the frame budget's phases as Tracy zones, behind a build feature
- **Area:** Tools and measurement
- **Cards:** all
- **Goal:** The runtime's frame budget puts every thread in exactly one named phase at a time (submission lock waits, refresh, textures, copy-back, the card's waits and more). Attach a Tracy zone to each phase, by the phase's stable name, behind a new `tracy` build feature that is off, so a contributor can see every thread's timeline in Tracy's viewer.
- **What is known, and how:** by reading the code: `crates/victory-gpu/src/budget.rs` says that `begin` and `end` are the single place a phase scope starts and stops, and that a Tracy zone attaches there by `NAMES`, "behind a feature that does not exist yet; nothing is emitted today".
- **Where the code is:** `crates/victory-gpu/src/budget.rs` (`Phase`, `NAMES`, `begin`, `end`); `crates/victory-gpu/Cargo.toml` (the new feature and the optional dependency); `crates/victory-gpu/src/events.rs` for how the runtime already names its own GPU work for outside tools.
- **Files owned:** `crates/victory-gpu/src/budget.rs`, `crates/victory-gpu/Cargo.toml`, `Cargo.lock`
- **Ruled out:** Tracy's crash handler and its code transfer: build the client with `default-features = false` and leave out `crash-handler` and `code-transfer`, because code transfer would copy the title's machine code into trace files (docs/DIAGNOSTICS.md, "Tracy").
- **Done when:** with `--features tracy` a run connects to the Tracy viewer of the version the pinned client speaks (a new row in the runtime's README says which), and each thread shows its phases as zones with frame marks at each present; without the feature the build and its output are unchanged; both clippy runs and the tests pass. A test checks that every `Phase` has a zone name and that the names are `NAMES`.
- **Test without the title:** the name test and the build without the feature need no copy; the timeline needs your own copy.
- **Judged by:** Tracy itself; for one frame, its zone times beside a system trace (WPA) of the same thread in the same window.
- **Out of scope:** lock tracking (the Rust client has none; a wrapper is later work); GPU zones; any change to the budget's figures.
- **Size:** M
- **Needs:** none to build; your own copy to see a timeline.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
