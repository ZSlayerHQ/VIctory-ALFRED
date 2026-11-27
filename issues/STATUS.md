# Issue board status

Every task here is open. The area and cards come from each issue; "Needs" says what the work takes beyond this
repository, so a contributor without a copy of the title can pick the tasks marked "none". The work a maintainer
holds is listed in [TASKS.md](../TASKS.md), "Held by a maintainer", and has no issue.

| Issue | Area | Cards | Needs |
|---|---|---|---|
| [11 — Bound LDS accesses by the allocation](11-lds-bound.md) | gpu | all | none for the fixture; your own copy and an NVIDIA card for the Aftermath run |
| [14 — Find the white-outs and the washed-out start](14-white-out.md) | gpu | all | your own copy of the title; a capture tool |
| [19 — Make the caption's GPU figure agree with PresentMon](19-caption-gpu-number.md) | tools | all | your own copy for the agreement check; PresentMon |
| [23 — A character's forearms render purple or as coloured noise](23-forearm-noise.md) | gpu | all | your own copy of the title; RenderDoc or PIX |
| [25 — An interior blows out to white, with hard-edged light slabs](25-interior-white-out.md) | gpu | all | your own copy of the title, with a save in an affected interior; a capture tool |
| [29 — A reporter batch can be dropped under sender-mutex contention](29-reporter-contention.md) | tools | all | none |
| [30 — The target-dump refusal can stall recording](30-dump-refusal-stall.md) | tools | all | none |
| [31 — Reserve a shared byte budget before mapping capture readbacks](31-capture-budget.md) | tools | all | none |
| [32 — Snapshot a repeated list's records after each execution](32-repeated-list-timing.md) | tools | all | none |
| [33 — Release the submission lock before waiting for a target dump](33-dump-lock-wait.md) | tools | all | none |
| [34 — Reporter startup and rejected-count races](34-reporter-startup.md) | tools | all | none |
| [35 — Keep the recorder's refusal logs off the list-state lock](35-sync-logging.md) | tools | all | none |
| [36 — Count a direct target dump as card work](36-direct-dump-clock.md) | tools | all | none |
| [37 — Bind caption updates to a window generation](37-caption-window-generation.md) | tools | all | none |
| [38 — Serialize present selection and the texture-state round trip for target dumps](38-present-dump-serialization.md) | tools | all | none |
| [39 — The dump switch's row must state the unsafe requirement](39-dump-unsafe-doc.md) | tools | all | none |
| [40 — Prove the sampler branch behind a section of EXEC](40-sampler-branch-proof.md) | gpu | all | none for the fixture; your own copy of the title to confirm the group in the warm-up |
| [41 — Lower the primitive-ordered preamble](41-primitive-ordered-preamble.md) | gpu | all | your own copy of the title, and a card that reports rasterizer-ordered views, to confirm |
| [42 — Name the wide-wave programs the paired form refuses](42-wide-wave-census.md) | gpu | nvidia | your own copy of the title for the census and the warm-up; the fixture needs none |
| [43 — Record the written range a mapped resource is unmapped with](43-unmap-written-range.md) | cpu | all | none |
| [44 — Keep the memory-type flags of the title's allocations](44-allocation-memory-type-flags.md) | cpu | all | none |
| [45 — Keep each DirectStorage read's name and tag](45-storage-read-name-tag.md) | cpu | all | none |
| [48 — Stress fixture for ordering between recording threads](48-recording-thread-order-fixture.md) | cpu | all | none |
| [49 — Census the temporal upscaler's inputs and its card time](49-upscaler-input-census.md) | gpu | all | your own copy of the title; PIX or Nsight |
| [50 — Name each pass and its render targets](50-pass-targets-census.md) | tools | all | none for the test; a copy of the title for the run line |
| [51 — DLSS capability probe test](51-dlss-capability-probe.md) | gpu | nvidia | none; an NVIDIA RTX GPU to see the positive case |
| [52 — Log the ray-tracing structure builder's calls](52-builder-call-logging.md) | gpu | all | your own copy of the title and a run that streams the world; the unit tests need none |
| [53 — DXR tier probe fixture](53-dxr-tier-probe.md) | gpu | all | none; a DXR-capable GPU to see the positive case |
| [54 — Note what the rehydrate call receives](54-rehydrate-inputs-note.md) | gpu | all | issue 52 merged; your own copy of the title and a run that streams the world |
| [55 — Make the load line's builder count truthful](55-load-line-builder-count.md) | tools | all | none; a runtime build to see a positive count |
| [56 — Answer --help before the recorder check](56-help-before-recorder.md) | tools | all | none |
| [57 — Rotate and compress old run logs](57-rotate-compress-logs.md) | tools | all | none |
| [58 — Paint the runtime's window black from the start](58-black-window-background.md) | cpu | all | none |
| [59 — Name the pipelines whose draws are dropped, in the stats file](59-dropped-draws-by-pipeline.md) | tools | all | none for the test; your own copy for the run |
| [60 — A pipeline drawn into a target of another format](60-pipeline-target-format.md) | gpu | all | none |
| [61 — A wide word compare equal to the scalar one](61-avx2-word-compare.md) | cpu | all | none; a processor with AVX2 (any that runs the title) |
| [62 — Tracy zones at the frame budget's phases](62-tracy-zones.md) | tools | all | none to build; your own copy to see a timeline |
| [63 — GPU crate tests that bound real elapsed time](63-tests-on-a-test-clock.md) | tools | all | none |
| [64 — A watch test that loses its address to another test](64-fixed-address-test.md) | cpu | all | none |
| [65 — A PIX capture recipe for the harness](65-pix-capture-recipe.md) | tools | all | PIX on Windows (docs/DIAGNOSTICS.md); your own copy for the real capture |
| [66 — Ghosting behind moving characters and objects](66-ghosting.md) | gpu | all | your own copy of the title; PIX or RenderDoc |
| [67 — Characters lit wrongly for single frames](67-lighting-flips.md) | gpu | all | your own copy of the title; PIX or RenderDoc |
| [68 — A dark band across hair](68-hair-band.md) | gpu | all | your own copy of the title; a capture tool; a second vendor's card helps |
| [69 — Shadowed walls darken after about 35 seconds at rest](69-walls-darken-at-rest.md) | gpu | all | your own copy of the title; a capture tool |
| [70 — The default build on an AMD card](70-amd-run-report.md) | tools | amd | your own copy of the title; an AMD RDNA2 or later card; PresentMon |
| [71 — Measure the off-by-default speed switches on your machine](71-speed-switches-ab.md) | cpu | all | your own copy of the title; PresentMon |
| [72 — Name the other refused programs](72-other-refused-programs.md) | gpu | all | your own copy of the title for the census |
