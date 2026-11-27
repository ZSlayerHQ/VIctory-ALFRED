# 40 — Prove the sampler branch behind a section of EXEC

- **Title:** Prove the sampler branch behind a section of EXEC
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** A group of pixel programs is refused because an immediate sampler sits behind a branch the proof does not accept: the branch is neither a proved constant-buffer flag nor the `s_cbranch_execz` of the sample's own EXEC section. Admit that one branch shape exactly, so that group translates without changing any other program.
- **What is known, and how:** the warm-up's refusal list (the runtime's own count, from a run with the maintainers' copy): a few dozen programs are refused with the paired 64-lane form on, its default; one group is pixel programs with the reason "an immediate sampler chosen by control flow behind a branch neither a proved flag nor the section of its sample's EXEC is not supported". The proof's accepted forms are written at the top of `sampler_choice.rs` (by reading the code).
- **Where the code is:** `crates/victory-shader/src/emit/sampler_choice.rs`: `Lower::immediate_choice`, `Lower::admit`, `Lower::flag_branch`, `Lower::run_of` and `Lower::hangs_on`; the refusal is `BEHIND`. The tests are beside it in `crates/victory-shader/src/emit/sampler_choice_tests.rs`.
- **Files owned:** `crates/victory-shader/src/emit/sampler_choice.rs`, `crates/victory-shader/src/emit/sampler_choice_tests.rs`
- **Ruled out:** the sampler pair itself (one clamped, one wrapped) and the pixel program form: the same form translates in the programs the module already admits.
- **Done when:** a synthetic fixture with the new branch shape fails before the change with `BEHIND` and passes after; `VICTORY_WARP=1 cargo test -p victory-shader --locked sampler_choice` passes; with your own copy, the warm-up refuses fewer programs than your own count before the change, by the size of that group, and no program carries that reason.
- **Test:** `sampler_choice_tests.rs`: a pixel program with the branch shape and a sample in the join; assert the `Why` before and the translated program after. Keep the existing `UNPROVED` cases refused.
- **Test without the title:** yes, the fixture; the warm-up count needs your own copy.
- **Out of scope:** the other refusal classes; descriptors chosen among loaded samplers; the 64-lane forms.
- **Size:** M
- **Needs:** none for the fixture; your own copy of the title to confirm the group in the warm-up.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
