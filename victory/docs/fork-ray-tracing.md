# Ray tracing in this fork: the support and what has been built

Status, 2026-11-25: **nothing is lit by ray tracing yet.** Most of the
machinery translates and is unit-tested; it has not yet run end to end in the
game. Everything here is off by default.

## What a guest's programs do

A guest's indirect light (probes, bounce, reflections) and the shadows of
its lamps and sun may be ray traced. Without them, anything the sun does not
hit directly is black: interiors, the shadowed side of a face, rooms lit only
by lamps.

No ray-tracing API is used:

- Traversal is ordinary compute-shader code, which reads title memory by
  address.
- The acceleration structures are in the console GPU's own node format, the
  public one of its hardware family (see `crates/victory-shader/src/bvh.rs`
  and the open-source driver it follows), not a host format.
- Structures are built and refit on the GPU by compute programs.
- Only the per-node test is hardware: the instructions
  `image_bvh64_intersect_ray` and `image_bvh_intersect_ray`. One call tests a
  ray against a box node's four children (returning the hit children, nearest
  first) or against a triangle node (returning the intersection's numerator,
  denominator and barycentric numerators).

So mapping this onto the host's own ray tracing (DXR) is not practical: the
layer would have to know the geometry behind structures the guest need not
describe to it, and the traversal loops live in the guest's own programs.
This fork runs the guest's builder and traversal as translated shaders and
emulates the one instruction.

## What this fork adds

1. **Compute programs with a spilled parameter table translate.** A console
   compute program whose root parameters need more dwords than its user
   registers hold reads them from a table in memory instead; the translator
   turns those reads back into parameters (`crates/victory-shader/src/spill.rs`).
   Added with it: root unordered-access views (translator and host binding),
   byte and 16-bit buffer accesses, and several control-flow shapes.
2. **Title memory by raw address from shaders.** A directory with one entry per
   64 KiB of title address space leads a shader to the layer's host copy of
   that memory; loads, stores, atomic add/sub and float min/max go through it.
   Before a dispatch the host makes ready the memory its root views and root
   constants name.
3. **The ray test, emulated in DXIL.** Four slab tests and a sorting network
   for box nodes; a watertight ray/triangle test for triangle nodes; both forms
   of the instruction. A plain Rust reference lives in
   `crates/victory-shader/src/bvh.rs`; tests run the emitted DXIL on the GPU
   over 311 ray/node cases and compare with the reference bit for bit.
4. **The trace and shading programs translate** (most of the trace programs, and
   the irradiance-cache shading program). This needed lane masks treated as
   plain numbers on some paths, and exact forms for programs written for
   64-lane waves running on a GPU with 32-lane waves.
5. **A safety rule on the host side.** The builder's stages run only while
   every builder pipeline a guest has created has a host pipeline; otherwise
   none run (`VICTORY_BVH_BUILDER=1` asks for it; `VICTORY_BVH_QUARANTINE=1`
   also keeps the builder's stores out of title memory, for a first test). A
   partly working builder wrote garbage bounds into title memory and crashed
   the title.

## Known gaps

- With `VICTORY_BVH_BUILDER=1` the builder still stays off in a real run: a few
  of its pipelines are refused by the host (they still ask for 64-lane waves on
  a 32-lane GPU) or do not translate.
- Five shadow-mask trace programs do not translate yet.
- An access to a title page that has no host copy is lost (a read gives zero, a
  store is dropped).
- Not publicly documented, and assumed: the instruction's rounding, the order
  of two boxes entered at the same distance, and the result for a ray exactly
  through an edge.
- Cost on the GPU is unmeasured.

## Where to look

- `crates/victory-shader/src/bvh.rs`: the reference for the node test.
- `crates/victory-shader/src/emit/lower.rs`: the lowering of the two
  instructions and of memory by address (`Lower::memory`).
- `crates/victory-shader/src/resource.rs`: address tracking for descriptors
  and raw addresses.
- `crates/victory-gpu/src/pipeline.rs`, `draw.rs`: the builder rule and the
  host side of root views and title memory.
