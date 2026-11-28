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

### A character's head is not drawn (some first starts of a build)

- **Seen:** the head's skin is missing for the whole session; the eyes, teeth and hair remain. It happens on some
  first starts of a build, while the card's pipelines are still being compiled, and not on later starts.
- **Cause (found 2026-11-28):** the title paints each character's skin once when a session loads. A paint draw whose
  pipeline was still being compiled was dropped, so the skin stayed empty for that session.
- **Until the fix is in:** close the program and start it again. The second start has its pipelines ready.

### With the title's ray-traced light on

- **Seen:** a strip of blocks along the bottom edge, and near geometry double-exposed with edges that step in blocks
  about 16 pixels across.
- **Suspected:** tiles whose light probes were never activated.

### The water's own light

- **Known:** two tiled water-light passes used to leave dark rectangles on the water on 32-lane cards. The base's
  forms for 64-lane programs fixed them, and the passes now run by default; not yet re-checked in a measured run
  here. `VICTORY_WATER_LIGHT=0` leaves them out on cards that run 32 lanes at a time, such as NVIDIA's; elsewhere the
  switch does nothing.

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

## Settings

### Choosing 1920 x 1080 stops the next start

- **Seen:** with `1920 x 1080` chosen on the F1 menu's Resolution row, the next start stops before the first frame:
  the title halts under the change that gives it that output.
- **Until it is fixed:** do not choose it. To recover, change the line `VICTORY_MENU_RES=1080` in
  `%LOCALAPPDATA%\victory\victory-settings.txt` to `VICTORY_MENU_RES=4k` (or delete it). The menu's saved choice goes
  over `--res`, so `victory play --res 4k` recovers only a start that was given `--res 1080` itself.

## Tools and tests

### Some of the base's own tests and lints fail

- **Known:** a handful of the base's tests fail or flake, and `cargo clippy -- -D warnings` reports findings the
  base came with. The public CI builds the runtime and reports both without failing on them until they are cleared.

### The runtime's own measures are glue, not judges

- **Known:** frame times come from PresentMon, a thread's time from a system trace, the card's time from a GPU
  trace. The figures the runtime prints about itself are for orientation only (`docs/DIAGNOSTICS.md`).

### Left out of this public copy

- MFG Unlocker, a third-party mod for DLSS frame generation on RTX 20 and 30 cards: not in this build. FSR 3 frame
  generation (2x) is the way on those cards.
- Nothing is downloaded. NVIDIA's DLSS and frame generation files are placed by hand: their `Streamline` folder goes
  into `nvidia-sdk` under `%LOCALAPPDATA%\victory`, and a start without them says so.
