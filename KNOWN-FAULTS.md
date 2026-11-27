# Known faults

Faults already found, each with what is seen, what is known of its cause, how that was established, and its task.
Treat an entry as settled: link its issue, say what your work adds, and spend your tokens on what is not here. A
cause marked "suspected" is not shown yet. "Held" means a maintainer holds the work (TASKS.md, "Held by a
maintainer"). Figures are from the maintainers' test scene at the larger console's profile on an NVIDIA card.

## Speed

### The frame is limited by one thread

- **Seen:** about 4 to 5 frames a second in the open world.
- **Cause:** the thread that submits each frame is busy for about two thirds of the frame, more than half of that
  inside Windows: reading title memory through the kernel (tens of thousands of calls a frame), changing and
  querying page protection, and the card's interrupt work landing on its processor. It also waits about a fifth of
  the frame for copy-back. The other cores are mostly idle.
- **Established by:** a system trace (WPR and WPA) with the camera still.
- **Task:** held (spreading the thread's work across cores); measurements on other machines: [71](issues/71-speed-switches-ab.md).

## Picture

### Faces: 16-bit integer values read as zero

- **Seen:** through a 16-bit unsigned integer view the translator reads zero for every value, through a signed one
  0 or 0xFFFF; normalised and float views read correctly.
- **Cause:** every view is read through one float-typed handle and narrowed as a float.
- **Established by:** six fixtures on the software adapter; the cause by reading the code. Open until a run:
  whether the title's face programs make these loads.
- **Task:** held (the fix, behind a switch).

### Black dashes along clothing seams, hems and silhouettes

- **Seen:** single black pixels and short dashes along seams, hems and silhouettes, and on some wooden edges.
- **Cause, suspected:** the notices that title memory changed are global queues drained by whichever heap's walk
  runs first, so another heap never learns its texture changed, and a texture uploaded before its bytes were final
  is not uploaded again.
- **Established by:** reading the code, and failing fixtures for the missed notice; a fix is merged behind
  `VICTORY_TEXTURE_RECEIPTS` (off) and has not been run in the game.
- **Task:** held (texture receipts, step two).

### Textures flash black for a moment while moving

- **Seen:** during a camera pan, a single frame with a whole surface black; in an earlier build, clouds missing
  for two frames.
- **Cause, suspected:** draws dropped while the driver compiles their pipeline: both moments fall where about 1,000
  draws a frame were dropped for want of a pipeline. A new build starts with no pipelines on disk.
- **Established by:** the runtime's own dropped-draw counter beside a recording; no pipeline is named yet.
- **Task:** [59](issues/59-dropped-draws-by-pipeline.md) names the pipelines; pipelines carried across builds and
  made ahead are held.

### A glass door's dirt layer drops out for a frame or two

- **Seen:** the dirt layer on a glass door flickers away for a frame or two.
- **Cause:** not known; the stats give dropped draws only as averages over seconds and name no pipeline, so it
  could not be tied to one.
- **Established by:** a live session with the runtime's stats.
- **Task:** [59](issues/59-dropped-draws-by-pipeline.md), then a skip of the named pipeline to confirm.

### Ghosting behind moving characters and objects

- **Seen:** moving characters leave a faint duplicate; a pale block trails a moving object reflected in water.
- **Cause, suspected:** the title's temporal blend at about 4 frames a second: the block follows the object as the
  camera sways (in some samples of a recording, not all). The motion vectors are not checked yet.
- **Established by:** recordings.
- **Task:** [66](issues/66-ghosting.md)

### Characters lit wrongly for single frames

- **Seen:** characters lit very differently for one frame, then normal again.
- **Cause:** not known; the next step is the buffer present only some frames.
- **Established by:** recordings.
- **Task:** [67](issues/67-lighting-flips.md)

### A character's forearms render purple or as coloured noise

- **Seen:** one character's forearms are purple; others show coloured noise where tattoo or decal layers sit over
  skin, while faces are right.
- **Cause:** not known; the skin, decal and tattoo programs translate and run.
- **Established by:** recordings and the warm-up's refusal list.
- **Task:** [23](issues/23-forearm-noise.md)

### An interior blows out to white, with hard-edged slabs of light

- **Seen:** in some interiors, characters blow out to white and hard-edged slabs of
  coloured light lie across surfaces.
- **Cause:** not known; the lighting programs translate and run.
- **Established by:** recordings.
- **Task:** [25](issues/25-interior-white-out.md)

### A dark band across hair

- **Seen:** a dark band runs across characters' hair.
- **Cause, suspected:** a 64-lane program behind the hair's mask, run in a 32-lane form.
- **Established by:** recordings on an NVIDIA card.
- **Task:** [68](issues/68-hair-band.md)

### Walls in shadow darken after about 35 seconds at rest

- **Seen:** with the camera still, shadowed walls darken after about 35 seconds.
- **Cause:** not known.
- **Established by:** recordings.
- **Task:** [69](issues/69-walls-darken-at-rest.md)

### The world's first 30 seconds are washed out, and white-outs

- **Seen:** the first 30 seconds or so look washed out; at times the whole screen goes white.
- **Cause:** not known.
- **Established by:** recordings.
- **Task:** [14](issues/14-white-out.md)

### Ray-traced lighting and reflections are not drawn

- **Seen:** no ray-traced reflections or bounce light; most dropped draws belong to the ray-tracing structure
  builder, which ships switched off, and the frame line counts them apart.
- **Cause:** the title's ray-tracing structures are not built for the host's ray tracing on any card yet.
- **Established by:** the runtime's frame line; the builder's scope by reading the code.
- **Task:** the build is held for the maintainers' decision; starter tasks [52](issues/52-builder-call-logging.md),
  [53](issues/53-dxr-tier-probe.md), [54](issues/54-rehydrate-inputs-note.md).

### A pipeline drawn into a target of another format

- **Seen:** nothing yet; a claim to verify.
- **Cause, suspected:** a host pipeline keeps the target formats the title declared, and the draw path does not
  compare them with the targets bound; Direct3D 12 calls such a draw invalid.
- **Established by:** reading the code, and another fork's fix for the same case.
- **Task:** [60](issues/60-pipeline-target-format.md)

## Shaders

### A few dozen programs are refused

- **Seen:** with the paired 64-lane form on (the default), the warm-up refuses a few dozen programs, in five
  groups: pixel programs with a sampler chosen behind a branch the proof does not accept; pixel programs needing a
  hardware input (primitive-ordered pixel shading); wide-wave compute programs the paired form does not
  take; compute programs that read a descriptor as data (the shadow rays); and a few one-off lane-mask cases. With
  the form off, more are refused. Your own warm-up's list is the baseline: note its count before and after a
  change.
- **Cause:** each refusal names its reason.
- **Established by:** the warm-up's refusal list on an NVIDIA card (the runtime's own count, matched by program id).
- **Tasks:** [40](issues/40-sampler-branch-proof.md), [41](issues/41-primitive-ordered-preamble.md),
  [42](issues/42-wide-wave-census.md), [72](issues/72-other-refused-programs.md)

### A linked mesh pipeline the card refuses loses its draw

- **Seen:** a draw is lost when the card refuses the linked mesh and pixel pipeline made for it.
- **Cause:** no fallback to the unlinked pair; the stages' interface is not checked before linking.
- **Established by:** a review of the code, then fixtures; a fix is built and in review.
- **Task:** held.

## Memory and ordering (all held)

Each was found by reading the code unless it says otherwise; a fix for each is held.

- **Title buffer copies run out of order:** the title's in-frame buffer copies can run out of order, so an earlier
  submission reads bytes a later one wrote. Confirmed by five failing tests on the path the title's lists take.
- **A title memory copy written while earlier readers run:** a step that may write title memory waits for
  submissions in the air only when they declare writes, so a copy can land under readers still running. Not
  reproduced yet.
- **Pages mapped while earlier submissions may still write there:** mapping on the title's call does not wait for
  submissions in the air, and a late store into memory the title freed faults. Shown by two fixtures.
- **The page opener can reopen memory the title has since re-protected or freed:** with the guard ledger off (the
  default), stale watch entries outlive a protect, free or decommit. Not reproduced yet.
- **The default card hash misses one two-block pattern of sign-bit changes:** a card write of that shape would
  never reach the title. Found by a review; the wide form (`VICTORY_HASH_WIDE=1`, off) does not have the gap.
- **Textures are made again in a still scene:** about 1,600 host textures a minute with the camera still. Counted
  by the runtime's own `VICTORY_TEXTURE_MADE` line; the cause is not known.

## Tools and tests

### The caption's GPU figure is not a load

- **Seen:** the caption's GPU busy figure once disagreed by a wide margin with an independent GPU clock in the same
  frame.
- **Cause:** it was built from the runtime's own waits for the card.
- **Established by:** the comparison; PresentMon is the judge.
- **Task:** [19](issues/19-caption-gpu-number.md)

### The runtime's own measures are glue, not judges

- **Seen:** an independent check found that the runtime's frame timings and GPU timestamps overlap and do not sum
  ("idle" is not idle), and that the kernel-call counters do not see the title's own protection calls in the
  default setup.
- **Established by:** an independent review reading the code against what each tool reports.
- **Task:** the counters' fixes are held; use the established tools (docs/DIAGNOSTICS.md) for any figure that
  decides something.

### A watch test fails now and then

- **Seen:** `write_watched_memory_reused_at_a_guarded_view_is_asked_about` failed once in 15 runs.
- **Cause:** the test frees an address and asks for it again; another test can take it first.
- **Established by:** reading the test.
- **Task:** [64](issues/64-fixed-address-test.md)
