# 66 — Ghosting behind moving characters and objects

- **Title:** Find whether the ghosting behind moving characters is the low frame rate or a wrong motion vector
- **Area:** GPU side: picture, shaders, the card
- **Cards:** all
- **Goal:** Moving characters leave a faint duplicate of themselves, and a pale block trails a moving object reflected in water. Find whether this is the title's temporal blend at a few frames a second, which would shrink as the frame rate rises, or motion vectors that do not match the movement, which would not.
- **What is known, and how:** seen on recordings of runs at about 4 frames a second. On one recording the pale block in the water followed the object above it as the camera swayed, in some samples and not in others: a trail, not a wrong reflection. Suspected: the temporal blend at this frame rate. Not checked: the motion vectors the skinning writes against those the blend reads.
- **Where the code is:** to be confirmed by a maintainer; the census of issue 49 names the velocity target and the blend's inputs.
- **Files owned:** none for the measurement; the fix's files, if a fault is found, agreed on the task.
- **Done when:** for one moving character, a capture gives the velocity the skinning pass writes and the value the temporal pass reads at the same pixel, and the comment says whether they agree (figures, not pictures); if they do not, a fix with a fixture follows.
- **Test without the title:** no: this needs your own copy; a fix's fixture would not.
- **Judged by:** PIX or RenderDoc for the two values; a recording for "fixed".
- **Out of scope:** raising the frame rate; the upscaler's replacement.
- **Size:** M
- **Needs:** your own copy of the title; PIX or RenderDoc.
- **Claim:** one `CHECK-IN` comment holds it; check in every 60 minutes (48 hours for `model: human`); the forms are in TASKS.md.
