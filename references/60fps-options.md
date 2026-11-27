# The 60 fps step: options, in order

Starts after the steady 30 fps world.

1. **Real 60 first.** Lift the title's own frame cap and keep its game logic correct at the new rate: physics,
   animation and timers must not speed up. For how this engine family does it, the reference is public
   documentation of the same engine on PC.
2. **Frame generation afterwards, as an add-on, never a substitute.** It adds latency and leaves game logic at the
   old rate, so it cannot stand in for step 1.
   - **In VIctory:** VIctory owns presentation, so it could feed the game's depth and motion vectors to FSR 3 frame
     generation or NVIDIA Streamline.
   - **In the driver:** NVIDIA Smooth Motion on RTX 50 cards, with no code.
3. **OptiScaler does not apply.** It swaps a PC game's upscaler calls (the DLSS, FSR and XeSS SDKs); the console
   build's upscaling is inside its own shaders.

Measure each step with the `world-run` skill's PresentMon capture: frame time, GPU busy and latency.
