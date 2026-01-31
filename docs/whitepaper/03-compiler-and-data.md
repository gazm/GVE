# 03 – Compiler and Data

**Source:** [Compiler Pipeline](../workflows/compiler-pipeline.md), [Data Specifications](../data/data-specifications.md)

---

## Compiler Pipeline Position

The compiler runs in **Layer 2 (The Architect)**. It consumes DNA (JSON) and produces `.gve_bin` binaries that are served to the Engine and Forge. Versioning for streaming is carried in message headers, not inside the binary. See [Engine API](../architecture/engine-api.md) for the WASM Binary Messaging Protocol.

---

## Pipeline Stages

1. **Math JIT (Tensor core)** — The CSG tree from JSON is compiled into a PyTorch computational graph: primitives and operations (union, subtract, intersect, smooth_union) and modifiers (twist, bend, etc.) become vectorized functions `(N,3) → (N,)` distance. This gives GPU-executable SDF math and automatic differentiation for later stages.
2. **Volume bake** — SDF is sampled into 3D textures for LOD 1 (mid-range) raymarching.
3. **Shell generation** — Dual Contouring produces the low-poly proxy mesh used for early-Z and raymarch bounds.
4. **Splat training** — Splats are placed and refined (e.g. zero-layer training) to snap to the SDF surface; exported for the splat pass.
5. **Binary export** — All artifacts are packed into `.gve_bin` (header, geometry, volume, splats, audio patch, etc.).

---

## Data Model: DNA (JSON)

Source assets are **DNA** JSON: a tree of primitives, CSG operations, and modifiers. Key concepts:

- **Primitives:** Sphere, Box, Cylinder, Capsule, Torus, Cone, Plane, plus advanced (polyhedron, Bezier tube, heightmap, splat clouds, foliage, etc.).
- **Operations:** union, subtract, intersect, smooth_union.
- **Variables (`$var_name`):** Root-level parameters; expressions like `"$length * 0.5 + 0.2"` are evaluated at compile time so one knob (e.g. caliber) can drive multiple fields.
- **Anchors (`@parent`, `@prev`, `@root`):** Relative positioning from bounding-box accessors (`.top`, `.bottom`, `.center`, `.left`, etc.) so the AI does not need absolute coordinates.

The compiler resolves variables and anchors to concrete numbers before baking; the runtime sees only binary data.

---

## Game Unit and Grid

Units are SI-based for consistency across physics, audio, and rendering:

- **1.0 GU = 1.0 m** (distance); gravity ≈ -9.8 GU/s².
- **1 Chunk = 4.0 GU**; Brickmap culling uses 4×4×4 m regions.
- **1.0 Mass = 1.0 kg**; used for inertia (e.g. volume × density).

---

## Binary Format (.gve_bin)

The runtime binary has a header (e.g. `GVEBinaryHeader`) and blocks for metadata, geometry (math bytecode or references), volume textures, splat data, and audio patch. Exact layout and offsets are defined in the Engine shared crate and documented in [Data Specifications](../data/data-specifications.md).

---

[Back to Whitepaper](README.md) | Prev: [02 – Rendering and Geometry](02-rendering-and-geometry.md) | Next: [04 – AI and Workflows](04-ai-and-workflows.md)
