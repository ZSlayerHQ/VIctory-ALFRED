# Known faults

Faults already found, each with what is seen and what is known of its cause. Treat an entry as settled: say what
your work adds, and spend your tokens on what is not here. A cause marked "suspected" is not shown yet. Figures are
from the maintainers' test scene at the larger console's profile and its 4K picture, on an NVIDIA RTX 5090, measured
with PresentMon on 2026-11-28.

**The runtime's base changed on 2026-11-28** (see `THIRD-PARTY.md`). This page describes the new base. Faults recorded
on the earlier one (ghosting behind moving characters, lighting that flips for single frames, textures that flash
black while moving, white-outs in the opening scene, dark bands across hair) have not been re-checked here: treat
them as open questions, not as known.

## Speed

### The frame is limited by one thread

- **Seen:** about 40 to 45 ms a frame (22 to 25 frames a second) in the open world. A build's first start is slower
  while its shaders are prepared.
- **Known:** by the runtime's own count, most of a frame goes to one thread that prepares the title's GPU work:
  copies between the memory the title shares with its GPU and the card's own memory.
- **Open:** a system trace of that thread, to say which waits and copies can go.

### The title's ray-traced light costs about 12 ms a frame

- **Seen:** about 54 to 56 ms a frame with `VICTORY_RT_RESOLVE=1` and `VICTORY_ACTIVATE_RAYS=1`.
- **Known:** the title traces its own rays in compute programs; here they run as ordinary compute work, and the
  extra time is on the submitting thread. Both switches are off by default, and a stand-in lights interiors.

## Picture

### Grain on the edges of sun shadows, hair and beards

- **Seen:** speckled edges where sunlight falls through leaves, and dithered hair and beard outlines. It comes and
  goes with the lighting: none in shade.
- **Suspected:** the title draws these with per-pixel noise and smooths them by blending several frames; that blend
  is not accumulating as it should.

### Red blocks flicker at the bottom and right edges of the screen

- **Seen:** solid red patches made of blocks about 16 pixels across, appearing in one frame and changing a block at
  a time. They stay fixed on the screen while the scene moves behind them.
- **Suspected:** the per-tile light lists are not written for the last row or column of tiles, which keep stale
  lists.

### At a distance a character's head is not drawn

- **Seen:** the head's skin is missing, the eyes remain, and the hair sits on a shoulder. Close up the head is right.
- **Suspected:** programs that prepare distant characters use all 64 lanes of a console wave; on 32-lane cards they
  run in an approximate form (see Shaders) that picks wrong values for half the lanes.

### With the title's ray-traced light on

- **Seen:** a strip of blocks along the bottom edge, and near geometry double-exposed with edges that step in blocks
  about 16 pixels across.
- **Suspected:** tiles whose light probes were never activated.

### The water's own light is switched off

- **Seen:** with it on, dark rectangles on the water.
- **Known:** two tiled water-light passes produce them; the base leaves both off by default.

## Shaders

### 64-lane programs on 32-lane cards

- **Known:** the console's GPU runs programs 64 lanes at a time; NVIDIA cards run 32. Each console wave is split in
  two halves here, and a program that reads across the whole wave is translated in an approximate form.
- **Switches:** `VICTORY_STRICT_HALVES=1` refuses those programs instead (their draws are dropped: characters lose
  their bodies). `VICTORY_WAVE_WIDE=1` is an exact last-resort form that is slower and can hang the card.

### Programs that do not translate

- **Seen:** about 225 of the title's roughly 25,800 programs do not translate at 32 lanes. The test scene uses one
  of them, a compute program that collects triangles of small mesh pieces; its draws are dropped.
- **How to count them:** the harness's run summary prints the programs a run could not translate.

## Tools and tests

### Some of the base's own tests and lints fail

- **Known:** a handful of the base's tests fail or flake, and `cargo clippy -- -D warnings` reports findings the
  base came with. The public CI builds the runtime and reports both without failing on them until they are cleared.

### The runtime's own measures are glue, not judges

- **Known:** frame times come from PresentMon, a thread's time from a system trace, the card's time from a GPU
  trace. The figures the runtime prints about itself are for orientation only (`docs/DIAGNOSTICS.md`).

### Left out of this public copy

- The base's 1440p output mode, DLSS and frame generation: they work by changing the title's own code, and files
  that do that are not published here.
