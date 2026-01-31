# 05 – Audio and Physics

**Source:** [System Overview](../architecture/overview.md) §4, [Audio System](../core-systems/audio-system.md), [Physics System](../core-systems/physics-system.md)

---

## Resonance Engine (Physics-Driven Audio)

Audio is synthesized from physical state instead of playing static samples. Collisions supply **velocity** and **Material_ID**; each `.gve_bin` carries an **AudioPatch** (FM carrier/modulator ratios, ADSR, DSP chain). The engine uses this to drive timbre and intensity so sound matches the object and impact.

**DSP path:**

1. **Source** — FM oscillator (carrier + modulator) for tonal body, or granular sampler for texture.
2. **Filter** — Low-pass (e.g. mass-driven) and distortion (e.g. material/rust) shape the signal.
3. **Mix** — Voice is sent to the master bus.
4. **Occlusion** — A raycast through the visual voxel volume determines whether sound is muffled by geometry; the same volume data used for rendering is reused for audio.

Fundamental frequency can be derived from stiffness and mass; modulation index from velocity. Edge cases (zero velocity, envelope complete, out-of-range f0) are handled by skipping allocation or clamping.

---

## Physics (Rapier3d and SDF Colliders)

The runtime uses **Rapier3d** for rigid-body simulation. Collision strategy is hybrid:

- **Primitives** — Boxes, spheres, cylinders, etc. map to native Rapier colliders; analytical distance, O(1). The compiler can simplify some CSG to a single primitive and use a standard collider.
- **SdfShape** — For complex or generative geometry, a custom shape evaluates SDF bytecode (same math as rendering). Broad phase uses a shell AABB; narrow phase uses the SDF so visual and physical boundaries stay aligned.

Rapier is pure Rust, so the same code runs on desktop and in WASM (Forge), giving deterministic behavior. SIMD (e.g. via `simba`) is used where applicable.

---

[Back to Whitepaper](README.md) | Prev: [04 – AI and Workflows](04-ai-and-workflows.md) | Next: [06 – Runtime and Tooling](06-runtime-and-tooling.md)
